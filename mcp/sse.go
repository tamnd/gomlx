// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// The SSE transport reaches a server that lives behind HTTP rather than as a
// local process. It is not a symmetric byte stream the way a pipe is: the client
// reads messages from a long-lived server-sent-events response and writes them by
// POSTing to a separate endpoint the server names at the start. This file hides
// that split behind an io.ReadWriteCloser so the same connection and session sit
// on top unchanged. The handshake is: open the event stream, read the first
// "endpoint" event to learn where to POST, then treat every later "message" event
// as an incoming JSON-RPC line.

// sseStream presents the SSE event stream and the POST endpoint as one
// io.ReadWriteCloser. Reads drain incoming message events; writes POST one
// JSON-RPC message each.
type sseStream struct {
	client  *http.Client
	postURL string

	pr   *io.PipeReader
	pw   *io.PipeWriter
	body io.ReadCloser
	stop context.CancelFunc

	closeOnce sync.Once
}

// Read returns bytes from the incoming message events.
func (s *sseStream) Read(p []byte) (int, error) { return s.pr.Read(p) }

// Write POSTs one JSON-RPC message to the server's endpoint. It relies on the
// codec writing exactly one message per call, which it does.
func (s *sseStream) Write(p []byte) (int, error) {
	req, err := http.NewRequest(http.MethodPost, s.postURL, bytes.NewReader(p))
	if err != nil {
		return 0, fmt.Errorf("mcp: build sse post: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("mcp: sse post: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return 0, fmt.Errorf("mcp: sse post returned %d", resp.StatusCode)
	}
	return len(p), nil
}

// Close stops the event stream and unblocks any pending read.
func (s *sseStream) Close() error {
	s.closeOnce.Do(func() {
		s.stop()
		_ = s.body.Close()
		_ = s.pw.Close()
	})
	return nil
}

// pump parses the event stream and feeds each message event into the read pipe as
// one newline-terminated line, which is exactly what the codec on the other side
// expects.
func (s *sseStream) pump(r *bufio.Reader) {
	for {
		event, data, err := readSSEEvent(r)
		if err != nil {
			_ = s.pw.CloseWithError(err)
			return
		}
		if (event == "" || event == "message") && data != "" {
			if _, err := s.pw.Write([]byte(data + "\n")); err != nil {
				return
			}
		}
	}
}

// startSSE opens the event stream, reads the endpoint event, and returns a stream
// ready for a connection. The passed context governs the lifetime of the event
// stream, so cancelling it tears the transport down.
func startSSE(ctx context.Context, cfg ServerConfig, client *http.Client) (*sseStream, error) {
	if cfg.URL == "" {
		return nil, fmt.Errorf("mcp: sse server %q has no url", cfg.Name)
	}
	streamCtx, cancel := context.WithCancel(ctx)

	req, err := http.NewRequestWithContext(streamCtx, http.MethodGet, cfg.URL, nil)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("mcp: build sse request for %q: %w", cfg.Name, err)
	}
	req.Header.Set("Accept", "text/event-stream")
	resp, err := client.Do(req)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("mcp: open sse stream for %q: %w", cfg.Name, err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		cancel()
		return nil, fmt.Errorf("mcp: sse stream for %q returned %d", cfg.Name, resp.StatusCode)
	}

	reader := bufio.NewReader(resp.Body)
	postURL, err := readEndpoint(reader, cfg.URL)
	if err != nil {
		resp.Body.Close()
		cancel()
		return nil, fmt.Errorf("mcp: sse endpoint for %q: %w", cfg.Name, err)
	}

	pr, pw := io.Pipe()
	stream := &sseStream{
		client:  client,
		postURL: postURL,
		pr:      pr,
		pw:      pw,
		body:    resp.Body,
		stop:    cancel,
	}
	go stream.pump(reader)
	return stream, nil
}

// readEndpoint reads events until the "endpoint" event arrives and resolves its
// data against the stream URL into an absolute POST URL.
func readEndpoint(r *bufio.Reader, streamURL string) (string, error) {
	base, err := url.Parse(streamURL)
	if err != nil {
		return "", err
	}
	for {
		event, data, err := readSSEEvent(r)
		if err != nil {
			return "", err
		}
		if event == "endpoint" {
			ref, err := url.Parse(strings.TrimSpace(data))
			if err != nil {
				return "", fmt.Errorf("bad endpoint %q: %w", data, err)
			}
			return base.ResolveReference(ref).String(), nil
		}
	}
}

// readSSEEvent reads one server-sent event, returning its type and joined data.
// An event ends at a blank line; comment lines and unknown fields are ignored.
func readSSEEvent(r *bufio.Reader) (event, data string, err error) {
	var dataLines []string
	for {
		line, readErr := r.ReadString('\n')
		if readErr != nil {
			if readErr == io.EOF && (event != "" || len(dataLines) > 0) {
				return event, strings.Join(dataLines, "\n"), nil
			}
			return "", "", readErr
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if event == "" && len(dataLines) == 0 {
				continue // a separator between events, nothing buffered yet
			}
			return event, strings.Join(dataLines, "\n"), nil
		}
		if strings.HasPrefix(line, ":") {
			continue // comment, often a keep-alive
		}
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			event = value
		case "data":
			dataLines = append(dataLines, value)
		}
	}
}

// DialSSE opens an SSE MCP server, completes the handshake, and returns a ready
// session. Closing the session tears down the event stream. When cfg supplies no
// client the default client is used.
func DialSSE(ctx context.Context, cfg ServerConfig, info ClientInfo, client *http.Client) (*Session, error) {
	if client == nil {
		client = http.DefaultClient
	}
	stream, err := startSSE(ctx, cfg, client)
	if err != nil {
		return nil, err
	}
	conn := NewConn(stream, nil)
	sess := NewSession(conn, cfg.Name, info)
	if err := sess.Initialize(ctx); err != nil {
		_ = sess.Close()
		return nil, err
	}
	return sess, nil
}
