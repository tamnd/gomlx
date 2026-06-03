// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"errors"
	"net"
	"slices"
	"testing"
)

// pipeDialer returns a dialer whose sessions talk to the shared fake server over
// an in-memory pipe, so the manager can be tested without spawning processes.
// Servers named in fail return a dial error instead.
func pipeDialer(fail map[string]bool) Dialer {
	return func(ctx context.Context, cfg ServerConfig, info ClientInfo) (*Session, error) {
		if fail[cfg.Name] {
			return nil, errors.New("connection refused")
		}
		clientEnd, serverEnd := net.Pipe()
		go serveFakeMCP(serverEnd, serverEnd)
		sess := NewSession(NewConn(clientEnd, nil), cfg.Name, info)
		if err := sess.Initialize(ctx); err != nil {
			return nil, err
		}
		return sess, nil
	}
}

func testManager(t *testing.T, cfg Config, fail map[string]bool) *Manager {
	t.Helper()
	m := newManagerDial(cfg, ClientInfo{Name: "gomlx", Version: "test"}, pipeDialer(fail))
	t.Cleanup(func() { m.Close() })
	return m
}

func twoServerConfig() Config {
	return Config{Servers: map[string]ServerConfig{
		"files": {Name: "files", Transport: TransportStdio, Enabled: true},
		"web":   {Name: "web", Transport: TransportStdio, Enabled: true},
	}}
}

func TestManagerConnectPoolsTools(t *testing.T) {
	m := testManager(t, twoServerConfig(), nil)
	if err := m.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Both servers offer read and write, namespaced per server.
	names := toolNames(m.Tools())
	want := []string{"files__read", "files__write", "web__read", "web__write"}
	slices.Sort(names)
	if !slices.Equal(names, want) {
		t.Fatalf("pooled tools=%v want %v", names, want)
	}
}

func TestManagerConnectRecordsStatus(t *testing.T) {
	m := testManager(t, twoServerConfig(), map[string]bool{"web": true})
	if err := m.Connect(context.Background()); err != nil {
		t.Fatalf("Connect should succeed when at least one server connects: %v", err)
	}

	statuses := m.Statuses()
	if len(statuses) != 2 {
		t.Fatalf("want 2 statuses, got %d", len(statuses))
	}
	byName := map[string]ServerStatus{}
	for _, s := range statuses {
		byName[s.Name] = s
	}
	if byName["files"].State != StateConnected || byName["files"].ToolsCount != 2 {
		t.Fatalf("files status wrong: %+v", byName["files"])
	}
	if byName["web"].State != StateError || byName["web"].Error == "" {
		t.Fatalf("web should be in error state with a message: %+v", byName["web"])
	}
}

func TestManagerConnectSkipsDisabled(t *testing.T) {
	cfg := Config{Servers: map[string]ServerConfig{
		"on":  {Name: "on", Transport: TransportStdio, Enabled: true},
		"off": {Name: "off", Transport: TransportStdio, Enabled: false},
	}}
	m := testManager(t, cfg, nil)
	if err := m.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	statuses := m.Statuses()
	if len(statuses) != 1 || statuses[0].Name != "on" {
		t.Fatalf("disabled server should not be connected: %v", statuses)
	}
}

func TestManagerConnectAllFailedIsError(t *testing.T) {
	m := testManager(t, twoServerConfig(), map[string]bool{"files": true, "web": true})
	if err := m.Connect(context.Background()); err == nil {
		t.Fatal("Connect should error when every enabled server fails")
	}
}

func TestManagerCallToolRoutes(t *testing.T) {
	m := testManager(t, twoServerConfig(), nil)
	if err := m.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	res, err := m.CallTool(context.Background(), "web__read", map[string]any{"path": "x"})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.Content != "line one\nline two" {
		t.Fatalf("routed call result=%q", res.Content)
	}
}

func TestManagerCallToolUnknownServer(t *testing.T) {
	m := testManager(t, twoServerConfig(), nil)
	if err := m.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if _, err := m.CallTool(context.Background(), "ghost__read", nil); err == nil {
		t.Fatal("calling a tool on an unknown server should fail")
	}
}

func TestManagerCallToolGateBlocksDangerousArgs(t *testing.T) {
	m := testManager(t, twoServerConfig(), nil)
	if err := m.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	// The tool itself is safe, but the argument reaches for a sensitive path, so
	// the gate refuses it before the server is contacted.
	if _, err := m.CallTool(context.Background(), "files__read", map[string]any{"path": "/etc/passwd"}); err == nil {
		t.Fatal("a dangerous argument should be refused by the gate")
	}
}

func TestManagerCloseDisconnects(t *testing.T) {
	m := testManager(t, twoServerConfig(), nil)
	if err := m.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	m.Close()
	// After close the registry is empty and a call has no server to route to.
	if got := m.Tools(); len(got) != 0 {
		t.Fatalf("tools should be cleared after close, got %v", got)
	}
	if _, err := m.CallTool(context.Background(), "files__read", nil); err == nil {
		t.Fatal("a call after close should fail")
	}
}
