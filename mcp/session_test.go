// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"net"
	"slices"
	"testing"
	"time"
)

// startServer runs the shared fake MCP server (see serveFakeMCP) over a pipe end.
func startServer(t *testing.T, serverEnd net.Conn) {
	t.Helper()
	go serveFakeMCP(serverEnd, serverEnd)
}

func newSession(t *testing.T) *Session {
	t.Helper()
	clientEnd, serverEnd := net.Pipe()
	startServer(t, serverEnd)
	conn := NewConn(clientEnd, nil)
	sess := NewSession(conn, "files", ClientInfo{Name: "gomlx", Version: "test"})
	t.Cleanup(func() { sess.Close() })
	return sess
}

func TestSessionInitialize(t *testing.T) {
	sess := newSession(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := sess.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if info := sess.ServerInfo(); info.Name != "fake" || info.Version != "1.0" {
		t.Fatalf("server info=%+v", info)
	}
}

func TestSessionListTools(t *testing.T) {
	sess := newSession(t)
	ctx := context.Background()
	if err := sess.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	tools, err := sess.ListTools(ctx)
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	names := []string{tools[0].FullName(), tools[1].FullName()}
	if !slices.Equal(names, []string{"files__read", "files__write"}) {
		t.Fatalf("tools not stamped with server name: %v", names)
	}
	if tools[0].Description != "read a file" {
		t.Fatalf("description lost: %q", tools[0].Description)
	}
}

func TestSessionCallToolJoinsText(t *testing.T) {
	sess := newSession(t)
	ctx := context.Background()
	if err := sess.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	res, err := sess.CallTool(ctx, "read", map[string]any{"path": "x"})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error result: %+v", res)
	}
	if res.Content != "line one\nline two" {
		t.Fatalf("text blocks not joined: %q", res.Content)
	}
}

func TestSessionCallToolErrorResult(t *testing.T) {
	sess := newSession(t)
	ctx := context.Background()
	if err := sess.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	// A tool that reports an error returns a result flagged as error, not a
	// transport error: the call reached the server and ran.
	res, err := sess.CallTool(ctx, "boom", nil)
	if err != nil {
		t.Fatalf("CallTool transport error: %v", err)
	}
	if !res.IsError || res.ErrorMessage != "it failed" {
		t.Fatalf("error result not surfaced: %+v", res)
	}
}

func TestSessionCallToolDefaultsArgs(t *testing.T) {
	sess := newSession(t)
	ctx := context.Background()
	if err := sess.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	// A nil argument map must still produce a valid call.
	if _, err := sess.CallTool(ctx, "read", nil); err != nil {
		t.Fatalf("CallTool with nil args: %v", err)
	}
}
