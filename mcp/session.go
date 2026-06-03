// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"fmt"
	"strings"
)

// A session is the MCP conversation that runs on top of a connection: the
// initialize handshake that agrees a protocol version, the tools/list call that
// discovers what a server offers, and the tools/call that runs one. It maps the
// protocol's wire shapes to and from the Tool and ToolResult types the rest of
// the package uses, and it stamps each discovered tool with the server's name so
// the registry can namespace it. The session does not care how the byte stream
// underneath was opened; it takes a Conn and speaks the protocol over it.

// ProtocolVersion is the MCP revision this client advertises in the handshake.
const ProtocolVersion = "2025-06-18"

// ClientInfo identifies this client to a server during initialization.
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ServerInfo is what a server reports about itself in the handshake.
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// Session is an initialized MCP conversation with one server. It is safe for
// concurrent use to the extent Conn is.
type Session struct {
	conn       *Conn
	serverName string
	clientInfo ClientInfo
	serverInfo ServerInfo
}

// NewSession wraps a connection as a session for the named server. serverName is
// used to namespace the tools this server offers; clientInfo is sent in the
// handshake.
func NewSession(conn *Conn, serverName string, clientInfo ClientInfo) *Session {
	return &Session{conn: conn, serverName: serverName, clientInfo: clientInfo}
}

// initializeResult is the server's reply to the initialize request.
type initializeResult struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ServerInfo      ServerInfo     `json:"serverInfo"`
}

// Initialize performs the MCP handshake: it sends the initialize request, records
// the server's identity, and sends the initialized notification that tells the
// server the client is ready. It must complete before tools are listed or called.
func (s *Session) Initialize(ctx context.Context) error {
	params := map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo":      s.clientInfo,
	}
	var res initializeResult
	if err := s.conn.Call(ctx, "initialize", params, &res); err != nil {
		return fmt.Errorf("mcp: initialize %q: %w", s.serverName, err)
	}
	s.serverInfo = res.ServerInfo
	if err := s.conn.Notify("notifications/initialized", map[string]any{}); err != nil {
		return fmt.Errorf("mcp: initialized notification %q: %w", s.serverName, err)
	}
	return nil
}

// ServerInfo returns what the server reported during initialization.
func (s *Session) ServerInfo() ServerInfo { return s.serverInfo }

// wireTool is the tool shape in a tools/list response.
type wireTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

// listToolsResult is the tools/list response.
type listToolsResult struct {
	Tools []wireTool `json:"tools"`
}

// ListTools asks the server for its tools and returns them stamped with this
// session's server name.
func (s *Session) ListTools(ctx context.Context) ([]Tool, error) {
	var res listToolsResult
	if err := s.conn.Call(ctx, "tools/list", map[string]any{}, &res); err != nil {
		return nil, fmt.Errorf("mcp: list tools %q: %w", s.serverName, err)
	}
	tools := make([]Tool, 0, len(res.Tools))
	for _, t := range res.Tools {
		tools = append(tools, Tool{
			ServerName:  s.serverName,
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	return tools, nil
}

// contentBlock is one block of a tool result's content.
type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// callToolResult is the tools/call response.
type callToolResult struct {
	Content []contentBlock `json:"content"`
	IsError bool           `json:"isError"`
}

// CallTool runs a tool on the server and returns its result. The protocol carries
// the result as a list of content blocks; the text blocks are joined into the
// result content, and an error result carries that text as its message.
func (s *Session) CallTool(ctx context.Context, name string, args map[string]any) (ToolResult, error) {
	if args == nil {
		args = map[string]any{}
	}
	params := map[string]any{"name": name, "arguments": args}
	var res callToolResult
	if err := s.conn.Call(ctx, "tools/call", params, &res); err != nil {
		return ToolResult{}, fmt.Errorf("mcp: call tool %q on %q: %w", name, s.serverName, err)
	}

	text := joinText(res.Content)
	result := ToolResult{ToolName: name, Content: text, IsError: res.IsError}
	if res.IsError {
		result.ErrorMessage = text
	}
	return result, nil
}

// Close shuts the underlying connection.
func (s *Session) Close() error { return s.conn.Close() }

// joinText concatenates the text of every text block, one per line.
func joinText(blocks []contentBlock) string {
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}
