// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

// The manager is the top of the MCP subsystem: it reads a config, connects to
// every enabled server, pools their tools in one registry, and routes a tool call
// to the session that owns it. One server failing to connect does not sink the
// rest, since a config often lists more servers than any one machine has running;
// the failure is recorded as that server's status and the others carry on. Tool
// calls go through the registry's security gate before they reach a session, so a
// blocked tool is refused here without a round trip.

// Dialer opens a ready, initialized session to a server. It is the seam the
// transports plug into, and the seam tests substitute to avoid real processes.
type Dialer func(ctx context.Context, cfg ServerConfig, info ClientInfo) (*Session, error)

// ServerStatus is a snapshot of one server's connection.
type ServerStatus struct {
	Name       string
	State      ServerState
	Transport  Transport
	ToolsCount int
	Error      string
}

// Manager owns the connections to every configured server and the registry their
// tools are pooled in. It is safe for concurrent use.
type Manager struct {
	cfg  Config
	info ClientInfo
	dial Dialer
	reg  *Registry

	mu       sync.Mutex
	sessions map[string]*Session
	status   map[string]ServerStatus
}

// NewManager returns a manager for cfg that identifies itself as info, using the
// default transport dialer.
func NewManager(cfg Config, info ClientInfo) *Manager {
	return newManagerDial(cfg, info, DefaultDial)
}

// newManagerDial is NewManager with an injectable dialer, used by tests.
func newManagerDial(cfg Config, info ClientInfo, dial Dialer) *Manager {
	return &Manager{
		cfg:      cfg,
		info:     info,
		dial:     dial,
		reg:      NewRegistry(cfg),
		sessions: make(map[string]*Session),
		status:   make(map[string]ServerStatus),
	}
}

// DefaultDial opens a session over the transport named in cfg. SSE is recognized
// but not yet implemented, and says so plainly rather than failing as if the
// server were down.
func DefaultDial(ctx context.Context, cfg ServerConfig, info ClientInfo) (*Session, error) {
	switch cfg.Transport {
	case TransportStdio, "":
		return DialStdio(ctx, cfg, info)
	case TransportSSE:
		return nil, fmt.Errorf("mcp: sse transport not yet supported for server %q", cfg.Name)
	default:
		return nil, fmt.Errorf("mcp: unknown transport %q for server %q", cfg.Transport, cfg.Name)
	}
}

// Registry returns the registry the manager pools tools in.
func (m *Manager) Registry() *Registry { return m.reg }

// Connect dials every enabled server in parallel, completes each handshake,
// lists its tools into the registry, and records its status. It returns an error
// only when every enabled server failed; a partial success is reported through
// the per-server statuses, not as an error.
func (m *Manager) Connect(ctx context.Context) error {
	type outcome struct {
		name   string
		status ServerStatus
		sess   *Session
	}

	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		outcomes []outcome
		enabled  int
	)
	for name, cfg := range m.cfg.Servers {
		if !cfg.Enabled {
			continue
		}
		enabled++
		wg.Add(1)
		go func(name string, cfg ServerConfig) {
			defer wg.Done()
			o := outcome{name: name}
			sess, tools, err := m.connectOne(ctx, cfg)
			if err != nil {
				o.status = ServerStatus{Name: name, State: StateError, Transport: cfg.Transport, Error: err.Error()}
			} else {
				o.sess = sess
				o.status = ServerStatus{Name: name, State: StateConnected, Transport: cfg.Transport, ToolsCount: len(tools)}
			}
			mu.Lock()
			outcomes = append(outcomes, o)
			mu.Unlock()
		}(name, cfg)
	}
	wg.Wait()

	connected := 0
	m.mu.Lock()
	for _, o := range outcomes {
		m.status[o.name] = o.status
		if o.sess != nil {
			m.sessions[o.name] = o.sess
			connected++
		}
	}
	m.mu.Unlock()

	if enabled > 0 && connected == 0 {
		return fmt.Errorf("mcp: no server connected out of %d enabled", enabled)
	}
	return nil
}

// connectOne dials a single server and reads its tool list into the registry.
func (m *Manager) connectOne(ctx context.Context, cfg ServerConfig) (*Session, []Tool, error) {
	sess, err := m.dial(ctx, cfg, m.info)
	if err != nil {
		return nil, nil, err
	}
	tools, err := sess.ListTools(ctx)
	if err != nil {
		_ = sess.Close()
		return nil, nil, fmt.Errorf("mcp: list tools from %q: %w", cfg.Name, err)
	}
	m.reg.SetTools(cfg.Name, tools)
	return sess, tools, nil
}

// Statuses returns a snapshot of every configured server's status, sorted by name.
func (m *Manager) Statuses() []ServerStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]ServerStatus, 0, len(m.status))
	for _, s := range m.status {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Tools returns every pooled tool that clears the security gate, in OpenAI format,
// ready to offer to the model.
func (m *Manager) Tools() []map[string]any { return m.reg.OpenAITools() }

// CallTool routes a call to the server that owns the named tool. The call passes
// through the registry's security gate first, so a high-risk tool or a dangerous
// argument is refused before any server is contacted. A tool that runs and reports
// a failure comes back as a ToolResult with IsError set, distinct from a routing
// or transport error.
func (m *Manager) CallTool(ctx context.Context, fullName string, args map[string]any) (ToolResult, error) {
	tool, err := m.reg.Resolve(fullName, args)
	if err != nil {
		return ToolResult{}, err
	}
	m.mu.Lock()
	sess, ok := m.sessions[tool.ServerName]
	m.mu.Unlock()
	if !ok {
		return ToolResult{}, fmt.Errorf("mcp: server %q for tool %q is not connected", tool.ServerName, fullName)
	}
	return sess.CallTool(ctx, tool.Name, args)
}

// Close disconnects every server and clears the registry.
func (m *Manager) Close() error {
	m.mu.Lock()
	sessions := m.sessions
	m.sessions = make(map[string]*Session)
	for name, s := range m.status {
		s.State = StateDisconnected
		m.status[name] = s
	}
	m.mu.Unlock()

	for name, sess := range sessions {
		_ = sess.Close()
		m.reg.RemoveServer(name)
	}
	return nil
}
