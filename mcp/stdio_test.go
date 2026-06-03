// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"slices"
	"testing"
	"time"
)

// When this environment variable is set, the test binary runs as a fake MCP
// server over stdin and stdout instead of running tests. The stdio transport test
// launches this same binary in that mode, which gives it a real subprocess that
// speaks the protocol without shipping a separate helper program.
const fakeServerEnv = "GOMLX_MCP_FAKE_SERVER"

func TestMain(m *testing.M) {
	if os.Getenv(fakeServerEnv) == "1" {
		serveFakeMCP(os.Stdin, os.Stdout)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// serveFakeMCP answers the three protocol methods a session uses, reading requests
// from r and writing replies to w until the stream ends. It is shared by the
// in-memory pipe tests and the subprocess transport test.
func serveFakeMCP(r io.Reader, w io.Writer) {
	cdc := newCodec(r, w)
	for {
		m, err := cdc.readMessage()
		if err != nil {
			return
		}
		if m.Method == "" || m.Method == "notifications/initialized" {
			continue
		}
		reply := map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(m.ID)}
		switch m.Method {
		case "initialize":
			reply["result"] = map[string]any{
				"protocolVersion": ProtocolVersion,
				"capabilities":    map[string]any{},
				"serverInfo":      map[string]any{"name": "fake", "version": "1.0"},
			}
		case "tools/list":
			reply["result"] = map[string]any{
				"tools": []any{
					map[string]any{
						"name":        "read",
						"description": "read a file",
						"inputSchema": map[string]any{"type": "object"},
					},
					map[string]any{
						"name":        "write",
						"description": "write a file",
					},
				},
			}
		case "tools/call":
			var p struct {
				Name string `json:"name"`
			}
			_ = json.Unmarshal(m.Params, &p)
			if p.Name == "boom" {
				reply["result"] = map[string]any{
					"content": []any{map[string]any{"type": "text", "text": "it failed"}},
					"isError": true,
				}
			} else {
				reply["result"] = map[string]any{
					"content": []any{
						map[string]any{"type": "text", "text": "line one"},
						map[string]any{"type": "text", "text": "line two"},
					},
				}
			}
		default:
			reply["error"] = map[string]any{"code": -32601, "message": "method not found"}
		}
		_ = cdc.writeValue(reply)
	}
}

func TestDialStdioRoundTrip(t *testing.T) {
	cfg := ServerConfig{
		Name:      "fake",
		Transport: TransportStdio,
		Command:   os.Args[0],
		Env:       map[string]string{fakeServerEnv: "1"},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	sess, err := DialStdio(ctx, cfg, ClientInfo{Name: "gomlx", Version: "test"})
	if err != nil {
		t.Fatalf("DialStdio: %v", err)
	}
	defer sess.Close()

	if info := sess.ServerInfo(); info.Name != "fake" {
		t.Fatalf("handshake did not complete over stdio: %+v", info)
	}

	tools, err := sess.ListTools(ctx)
	if err != nil {
		t.Fatalf("ListTools over stdio: %v", err)
	}
	names := []string{tools[0].FullName(), tools[1].FullName()}
	if !slices.Equal(names, []string{"fake__read", "fake__write"}) {
		t.Fatalf("tools over stdio wrong: %v", names)
	}

	res, err := sess.CallTool(ctx, "read", map[string]any{"path": "x"})
	if err != nil {
		t.Fatalf("CallTool over stdio: %v", err)
	}
	if res.Content != "line one\nline two" {
		t.Fatalf("tool result over stdio: %q", res.Content)
	}
}

func TestDialStdioCloseStopsProcess(t *testing.T) {
	cfg := ServerConfig{
		Name:    "fake",
		Command: os.Args[0],
		Env:     map[string]string{fakeServerEnv: "1"},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	sess, err := DialStdio(ctx, cfg, ClientInfo{Name: "gomlx", Version: "test"})
	if err != nil {
		t.Fatalf("DialStdio: %v", err)
	}
	// Closing the session closes stdin, which our fake server treats as its cue
	// to exit; Close should return without hitting the kill grace period.
	done := make(chan error, 1)
	go func() { done <- sess.Close() }()
	select {
	case <-done:
	case <-time.After(stdioStreamKillGrace + 2*time.Second):
		t.Fatal("Close did not return; the subprocess was not shut down")
	}
}

func TestDialStdioBadCommand(t *testing.T) {
	cfg := ServerConfig{Name: "nope", Command: "this-command-does-not-exist-gomlx"}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := DialStdio(ctx, cfg, ClientInfo{}); err == nil {
		t.Fatal("dialing a missing command should fail")
	}
}

func TestMergeEnvOverridesAndPreserves(t *testing.T) {
	base := []string{"PATH=/bin", "HOME=/root"}
	out := mergeEnv(base, map[string]string{"HOME": "/home/u", "TOKEN": "abc"})
	want := []string{"HOME=/home/u", "PATH=/bin", "TOKEN=abc"}
	if !slices.Equal(out, want) {
		t.Fatalf("mergeEnv=%v want %v", out, want)
	}
	// No overrides returns the base untouched.
	if got := mergeEnv(base, nil); !slices.Equal(got, base) {
		t.Fatalf("nil overrides should return base, got %v", got)
	}
}
