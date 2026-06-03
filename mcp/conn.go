// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
)

// A JSON-RPC connection is full duplex: responses to the requests we send arrive
// interleaved with notifications the server pushes on its own. Conn runs a single
// read loop that sorts the two apart, routing each response back to the call that
// is waiting for it by matching the id, and handing every notification to a
// callback. Callers see a plain synchronous Call, with the concurrency handled
// here. The transports in this package supply the byte stream; how that stream is
// established, a subprocess or an HTTP connection, is their concern, not Conn's.

// ErrConnClosed is returned by Call once the connection has been closed or its
// read loop has stopped.
var ErrConnClosed = errors.New("mcp: connection closed")

// NotificationHandler receives a server-initiated notification. params is the raw
// JSON of the notification's params, which may be empty.
type NotificationHandler func(method string, params json.RawMessage)

// Conn is a JSON-RPC 2.0 connection over a byte stream. It is safe for concurrent
// use; many goroutines may Call at once.
type Conn struct {
	codec  *codec
	closer io.Closer
	onNote NotificationHandler

	mu       sync.Mutex
	nextID   int64
	pending  map[int64]chan callResult
	closed   bool
	closeErr error

	done chan struct{}
}

type callResult struct {
	resp *Response
	err  error
}

// NewConn starts a connection over rwc, calling onNote for each notification the
// peer sends. A nil onNote drops notifications. The read loop runs until rwc
// reaches EOF or Close is called.
func NewConn(rwc io.ReadWriteCloser, onNote NotificationHandler) *Conn {
	c := &Conn{
		codec:   newCodec(rwc, rwc),
		closer:  rwc,
		onNote:  onNote,
		pending: make(map[int64]chan callResult),
		done:    make(chan struct{}),
	}
	go c.readLoop()
	return c
}

// Call sends a request and waits for its response. It unmarshals a successful
// result into result when result is non-nil, returns the server's *RPCError when
// the response carries one, and respects ctx for cancellation.
func (c *Conn) Call(ctx context.Context, method string, params, result any) error {
	c.mu.Lock()
	if c.closed {
		err := c.closeErr
		c.mu.Unlock()
		if err == nil {
			err = ErrConnClosed
		}
		return err
	}
	id := c.nextID
	c.nextID++
	ch := make(chan callResult, 1)
	c.pending[id] = ch
	c.mu.Unlock()

	req := Request{JSONRPC: JSONRPCVersion, ID: id, Method: method, Params: params}
	if err := c.codec.writeValue(req); err != nil {
		c.discard(id)
		return err
	}

	select {
	case <-ctx.Done():
		c.discard(id)
		return ctx.Err()
	case <-c.done:
		return c.closeReason()
	case res := <-ch:
		if res.err != nil {
			return res.err
		}
		if res.resp.Error != nil {
			return res.resp.Error
		}
		if result != nil && len(res.resp.Result) > 0 {
			if err := json.Unmarshal(res.resp.Result, result); err != nil {
				return fmt.Errorf("mcp: decode result for %q: %w", method, err)
			}
		}
		return nil
	}
}

// Notify sends a notification, which expects no response.
func (c *Conn) Notify(method string, params any) error {
	c.mu.Lock()
	closed := c.closed
	c.mu.Unlock()
	if closed {
		return ErrConnClosed
	}
	return c.codec.writeValue(Notification{JSONRPC: JSONRPCVersion, Method: method, Params: params})
}

// Close shuts the connection and fails every in-flight and future Call.
func (c *Conn) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	if c.closeErr == nil {
		c.closeErr = ErrConnClosed
	}
	c.mu.Unlock()
	err := c.closer.Close()
	c.failPending(c.closeErr)
	return err
}

// readLoop reads messages until the stream ends, routing responses to waiting
// calls and notifications to the handler.
func (c *Conn) readLoop() {
	defer close(c.done)
	for {
		msg, err := c.codec.readMessage()
		if err != nil {
			c.shutdown(err)
			return
		}
		if msg.isResponse() {
			c.deliver(msg)
			continue
		}
		// A message with a method and no id is a notification; a server-initiated
		// request also has a method, but this client does not serve requests, so
		// both are surfaced through the notification handler.
		if msg.Method != "" && c.onNote != nil {
			c.onNote(msg.Method, msg.Params)
		}
	}
}

// deliver routes a response to the call waiting on its id.
func (c *Conn) deliver(msg message) {
	var id int64
	if err := json.Unmarshal(msg.ID, &id); err != nil {
		return // a response whose id is not one we issued is ignored
	}
	c.mu.Lock()
	ch, ok := c.pending[id]
	delete(c.pending, id)
	c.mu.Unlock()
	if !ok {
		return
	}
	resp := &Response{JSONRPC: msg.JSONRPC, Result: msg.Result, Error: msg.Error}
	ch <- callResult{resp: resp}
}

// shutdown records why the read loop stopped and fails all pending calls. A clean
// EOF is reported as ErrConnClosed.
func (c *Conn) shutdown(err error) {
	if err == io.EOF {
		err = ErrConnClosed
	}
	c.mu.Lock()
	if !c.closed {
		c.closed = true
		c.closeErr = err
	}
	c.mu.Unlock()
	c.failPending(err)
}

// failPending resolves every waiting call with err and clears the pending map.
func (c *Conn) failPending(err error) {
	c.mu.Lock()
	pending := c.pending
	c.pending = make(map[int64]chan callResult)
	c.mu.Unlock()
	for _, ch := range pending {
		ch <- callResult{err: err}
	}
}

// discard removes a pending call that will never be answered, as on cancellation.
func (c *Conn) discard(id int64) {
	c.mu.Lock()
	delete(c.pending, id)
	c.mu.Unlock()
}

// closeReason returns why the connection closed, defaulting to ErrConnClosed.
func (c *Conn) closeReason() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closeErr != nil {
		return c.closeErr
	}
	return ErrConnClosed
}
