// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
)

// MCP speaks JSON-RPC 2.0, and over a stdio transport each message is a single
// line of compact JSON terminated by a newline. This file holds the message types
// and the line codec both transports share. The codec only frames and parses
// messages; correlating a response to the request that asked for it lives in the
// connection in conn.go.

// JSONRPCVersion is the protocol version string carried in every message.
const JSONRPCVersion = "2.0"

// Request is a JSON-RPC request that expects a response. ID is a number or string
// the peer echoes back so the response can be matched to the request.
type Request struct {
	JSONRPC string `json:"jsonrpc"`
	ID      any    `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// Notification is a JSON-RPC message that expects no response. It carries no ID.
type Notification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// Response is a JSON-RPC response. Exactly one of Result or Error is set.
type Response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// RPCError is the error object a failed response carries.
type RPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *RPCError) Error() string {
	return fmt.Sprintf("jsonrpc error %d: %s", e.Code, e.Message)
}

// message is the wire envelope used to classify an incoming line. A request and a
// notification both carry a method; only a request and a response carry an id.
type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// codec frames JSON-RPC messages as newline-delimited compact JSON over an
// underlying reader and writer.
type codec struct {
	r *bufio.Reader
	w io.Writer
}

func newCodec(r io.Reader, w io.Writer) *codec {
	return &codec{r: bufio.NewReader(r), w: w}
}

// writeValue marshals v to compact JSON and writes it followed by a newline. A
// JSON-RPC line must not contain an embedded newline, which compact marshaling
// guarantees.
func (c *codec) writeValue(v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("mcp: marshal message: %w", err)
	}
	raw = append(raw, '\n')
	if _, err := c.w.Write(raw); err != nil {
		return fmt.Errorf("mcp: write message: %w", err)
	}
	return nil
}

// readMessage reads one line and parses it into an envelope. It returns io.EOF
// when the stream ends cleanly between messages.
func (c *codec) readMessage() (message, error) {
	line, err := c.r.ReadBytes('\n')
	if err != nil {
		// A final message without a trailing newline is still valid.
		if err == io.EOF && len(line) > 0 {
			var m message
			if uerr := json.Unmarshal(line, &m); uerr != nil {
				return message{}, fmt.Errorf("mcp: parse message: %w", uerr)
			}
			return m, nil
		}
		return message{}, err
	}
	var m message
	if uerr := json.Unmarshal(line, &m); uerr != nil {
		return message{}, fmt.Errorf("mcp: parse message: %w", uerr)
	}
	return m, nil
}

// isResponse reports whether the envelope is a response to a request, as opposed
// to a server-initiated request or notification. A response carries an id and no
// method.
func (m message) isResponse() bool {
	return len(m.ID) > 0 && m.Method == ""
}
