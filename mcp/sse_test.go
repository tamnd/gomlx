// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"
)

// sseTestServer is a minimal MCP SSE server. A GET opens the event stream; each
// request POSTed to the message endpoint is answered by writing a message event
// back on that stream, which is how the real transport delivers responses. The
// stream handler is the only goroutine that touches the response writer, fed by a
// channel, since net/http forbids concurrent writes to one writer.
type sseTestServer struct {
	*httptest.Server
	events chan []byte
}

func newSSETestServer() *sseTestServer {
	s := &sseTestServer{events: make(chan []byte, 16)}
	mux := http.NewServeMux()
	mux.HandleFunc("/sse", s.handleStream)
	mux.HandleFunc("/messages", s.handleMessage)
	s.Server = httptest.NewServer(mux)
	return s
}

func (s *sseTestServer) handleStream(w http.ResponseWriter, r *http.Request) {
	flush, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "no flush", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)

	// Announce where to POST messages, then write each reply as it is produced
	// until the client goes away. This goroutine owns the writer.
	_, _ = w.Write([]byte("event: endpoint\ndata: /messages\n\n"))
	flush.Flush()
	for {
		select {
		case <-r.Context().Done():
			return
		case raw := <-s.events:
			_, _ = w.Write([]byte("event: message\ndata: " + string(raw) + "\n\n"))
			flush.Flush()
		}
	}
}

func (s *sseTestServer) handleMessage(w http.ResponseWriter, r *http.Request) {
	var m message
	if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusAccepted)

	if reply, send := fakeReply(m); send {
		raw, _ := json.Marshal(reply)
		s.events <- raw
	}
}

func TestDialSSERoundTrip(t *testing.T) {
	srv := newSSETestServer()
	defer srv.Close()

	cfg := ServerConfig{Name: "web", Transport: TransportSSE, URL: srv.URL + "/sse"}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	sess, err := DialSSE(ctx, cfg, ClientInfo{Name: "gomlx", Version: "test"}, srv.Client())
	if err != nil {
		t.Fatalf("DialSSE: %v", err)
	}
	defer sess.Close()

	if info := sess.ServerInfo(); info.Name != "fake" {
		t.Fatalf("handshake did not complete over sse: %+v", info)
	}

	tools, err := sess.ListTools(ctx)
	if err != nil {
		t.Fatalf("ListTools over sse: %v", err)
	}
	names := []string{tools[0].FullName(), tools[1].FullName()}
	if !slices.Equal(names, []string{"web__read", "web__write"}) {
		t.Fatalf("tools over sse wrong: %v", names)
	}

	res, err := sess.CallTool(ctx, "read", map[string]any{"path": "x"})
	if err != nil {
		t.Fatalf("CallTool over sse: %v", err)
	}
	if res.Content != "line one\nline two" {
		t.Fatalf("tool result over sse: %q", res.Content)
	}
}

func TestDialSSEBadURL(t *testing.T) {
	cfg := ServerConfig{Name: "web", Transport: TransportSSE, URL: "http://127.0.0.1:1/sse"}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := DialSSE(ctx, cfg, ClientInfo{}, nil); err == nil {
		t.Fatal("dialing an unreachable sse server should fail")
	}
}

func TestReadEndpointResolvesRelative(t *testing.T) {
	stream := "event: endpoint\ndata: /messages?sid=abc\n\n"
	r := bufio.NewReader(strings.NewReader(stream))
	got, err := readEndpoint(r, "http://host:8080/sse")
	if err != nil {
		t.Fatalf("readEndpoint: %v", err)
	}
	if got != "http://host:8080/messages?sid=abc" {
		t.Fatalf("resolved endpoint=%q", got)
	}
}

func TestReadSSEEventJoinsDataLines(t *testing.T) {
	r := bufio.NewReader(strings.NewReader("event: message\ndata: line1\ndata: line2\n\n"))
	event, data, err := readSSEEvent(r)
	if err != nil {
		t.Fatalf("readSSEEvent: %v", err)
	}
	if event != "message" || data != "line1\nline2" {
		t.Fatalf("event=%q data=%q", event, data)
	}
}

func TestReadSSEEventSkipsComments(t *testing.T) {
	// A leading comment line (keep-alive) before the real event must be ignored.
	r := bufio.NewReader(strings.NewReader(": keep-alive\nevent: endpoint\ndata: /m\n\n"))
	event, data, err := readSSEEvent(r)
	if err != nil {
		t.Fatalf("readSSEEvent: %v", err)
	}
	if event != "endpoint" || data != "/m" {
		t.Fatalf("comment not skipped: event=%q data=%q", event, data)
	}
}
