// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"sync"
	"testing"
	"time"
)

// fakeServer reads JSON-RPC lines from a pipe end and lets a test script react to
// each request. It mirrors how a real MCP server would behave without spawning a
// process or opening a socket.
type fakeServer struct {
	rw      io.ReadWriteCloser
	r       *bufio.Reader
	handler func(req message) // called for each request the client sends
}

func newFakeServer(rw io.ReadWriteCloser, handler func(message)) *fakeServer {
	s := &fakeServer{rw: rw, r: bufio.NewReader(rw), handler: handler}
	go s.loop()
	return s
}

func (s *fakeServer) loop() {
	for {
		line, err := s.r.ReadBytes('\n')
		if err != nil {
			return
		}
		var m message
		if json.Unmarshal(line, &m) == nil && m.Method != "" {
			s.handler(m)
		}
	}
}

func (s *fakeServer) send(v any) {
	raw, _ := json.Marshal(v)
	raw = append(raw, '\n')
	_, _ = s.rw.Write(raw)
}

func newConnPair(t *testing.T, handler func(*fakeServer, message)) (*Conn, chan Notification) {
	t.Helper()
	clientEnd, serverEnd := net.Pipe()
	var srv *fakeServer
	srv = newFakeServer(serverEnd, func(m message) { handler(srv, m) })

	notes := make(chan Notification, 8)
	conn := NewConn(clientEnd, func(method string, params json.RawMessage) {
		notes <- Notification{Method: method, Params: params}
	})
	t.Cleanup(func() { conn.Close() })
	return conn, notes
}

func TestConnCallReturnsResult(t *testing.T) {
	conn, _ := newConnPair(t, func(s *fakeServer, m message) {
		// Echo the method name back as the result.
		s.send(map[string]any{
			"jsonrpc": "2.0",
			"id":      json.RawMessage(m.ID),
			"result":  map[string]any{"method": m.Method},
		})
	})

	var out struct {
		Method string `json:"method"`
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := conn.Call(ctx, "tools/list", nil, &out); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if out.Method != "tools/list" {
		t.Fatalf("result=%q", out.Method)
	}
}

func TestConnCallSurfacesRPCError(t *testing.T) {
	conn, _ := newConnPair(t, func(s *fakeServer, m message) {
		s.send(map[string]any{
			"jsonrpc": "2.0",
			"id":      json.RawMessage(m.ID),
			"error":   map[string]any{"code": -32601, "message": "method not found"},
		})
	})

	err := conn.Call(context.Background(), "nope", nil, nil)
	var rpcErr *RPCError
	if !errors.As(err, &rpcErr) {
		t.Fatalf("want *RPCError, got %v", err)
	}
	if rpcErr.Code != -32601 {
		t.Fatalf("code=%d", rpcErr.Code)
	}
}

func TestConnConcurrentCallsCorrelate(t *testing.T) {
	// The server replies out of order to prove responses are matched by id, not
	// by arrival order.
	conn, _ := newConnPair(t, func(s *fakeServer, m message) {
		var p struct {
			N int `json:"n"`
		}
		_ = json.Unmarshal(m.Params, &p)
		go func() {
			// Reply to the lower n later, scrambling order.
			if p.N == 0 {
				time.Sleep(50 * time.Millisecond)
			}
			s.send(map[string]any{
				"jsonrpc": "2.0",
				"id":      json.RawMessage(m.ID),
				"result":  map[string]any{"n": p.N},
			})
		}()
	})

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			var out struct {
				N int `json:"n"`
			}
			if err := conn.Call(context.Background(), "echo", map[string]any{"n": n}, &out); err != nil {
				t.Errorf("call %d: %v", n, err)
				return
			}
			if out.N != n {
				t.Errorf("call %d got n=%d (responses mismatched)", n, out.N)
			}
		}(i)
	}
	wg.Wait()
}

func TestConnDeliversNotifications(t *testing.T) {
	conn, notes := newConnPair(t, func(s *fakeServer, m message) {
		// On any request, push a notification then answer.
		s.send(map[string]any{
			"jsonrpc": "2.0",
			"method":  "notifications/message",
			"params":  map[string]any{"level": "info"},
		})
		s.send(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(m.ID), "result": map[string]any{}})
	})

	if err := conn.Call(context.Background(), "ping", nil, nil); err != nil {
		t.Fatalf("Call: %v", err)
	}
	select {
	case n := <-notes:
		if n.Method != "notifications/message" {
			t.Fatalf("notification method=%q", n.Method)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("expected a notification")
	}
}

func TestConnCallRespectsContext(t *testing.T) {
	// The server never answers, so the call must end on context cancellation.
	conn, _ := newConnPair(t, func(s *fakeServer, m message) {})
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	err := conn.Call(ctx, "hang", nil, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want deadline exceeded, got %v", err)
	}
}

func TestConnCallAfterCloseFails(t *testing.T) {
	conn, _ := newConnPair(t, func(s *fakeServer, m message) {})
	conn.Close()
	if err := conn.Call(context.Background(), "x", nil, nil); err == nil {
		t.Fatal("a call after close should fail")
	}
}

func TestConnCloseUnblocksPendingCall(t *testing.T) {
	conn, _ := newConnPair(t, func(s *fakeServer, m message) {})
	errCh := make(chan error, 1)
	go func() {
		errCh <- conn.Call(context.Background(), "hang", nil, nil)
	}()
	time.Sleep(50 * time.Millisecond)
	conn.Close()
	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("a pending call should fail when the connection closes")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("close did not unblock the pending call")
	}
}
