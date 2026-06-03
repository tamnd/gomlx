// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
)

// These handlers expose the MCP subsystem over HTTP: what tools are pooled, what
// state each configured server is in, and a way to run one tool directly. They
// are the inspection and manual-invocation surface; the chat path reaches the
// same tools through the manager without going over HTTP. When the server was
// started without an MCP config the manager is nil, and the listing routes
// report an empty subsystem rather than failing, so a client can always ask.

// mcpToolInfo describes one pooled tool.
type mcpToolInfo struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Server      string         `json:"server"`
	Parameters  map[string]any `json:"parameters"`
}

// mcpToolsResponse is the body of GET /v1/mcp/tools.
type mcpToolsResponse struct {
	Tools []mcpToolInfo `json:"tools"`
	Count int           `json:"count"`
}

// mcpServerInfo is one server's connection snapshot.
type mcpServerInfo struct {
	Name       string `json:"name"`
	State      string `json:"state"`
	Transport  string `json:"transport"`
	ToolsCount int    `json:"tools_count"`
	Error      string `json:"error,omitempty"`
}

// mcpServersResponse is the body of GET /v1/mcp/servers.
type mcpServersResponse struct {
	Servers []mcpServerInfo `json:"servers"`
}

// mcpExecuteRequest is the body of POST /v1/mcp/execute.
type mcpExecuteRequest struct {
	ToolName  string         `json:"tool_name"`
	Arguments map[string]any `json:"arguments"`
}

// mcpExecuteResponse is the outcome of running a tool.
type mcpExecuteResponse struct {
	ToolName     string `json:"tool_name"`
	Content      any    `json:"content,omitempty"`
	IsError      bool   `json:"is_error"`
	ErrorMessage string `json:"error_message,omitempty"`
}

// MCPTools handles GET /v1/mcp/tools, listing every pooled tool by its
// namespaced name along with the server that owns it.
func (d *Deps) MCPTools(w http.ResponseWriter, r *http.Request) {
	if d.MCP == nil {
		writeJSON(w, http.StatusOK, mcpToolsResponse{Tools: []mcpToolInfo{}})
		return
	}
	tools := d.MCP.Registry().Tools()
	out := make([]mcpToolInfo, 0, len(tools))
	for _, t := range tools {
		out = append(out, mcpToolInfo{
			Name:        t.FullName(),
			Description: t.Description,
			Server:      t.ServerName,
			Parameters:  t.InputSchema,
		})
	}
	writeJSON(w, http.StatusOK, mcpToolsResponse{Tools: out, Count: len(out)})
}

// MCPServers handles GET /v1/mcp/servers, reporting each configured server's
// state, transport, tool count, and last error.
func (d *Deps) MCPServers(w http.ResponseWriter, r *http.Request) {
	if d.MCP == nil {
		writeJSON(w, http.StatusOK, mcpServersResponse{Servers: []mcpServerInfo{}})
		return
	}
	statuses := d.MCP.Statuses()
	out := make([]mcpServerInfo, 0, len(statuses))
	for _, s := range statuses {
		out = append(out, mcpServerInfo{
			Name:       s.Name,
			State:      string(s.State),
			Transport:  string(s.Transport),
			ToolsCount: s.ToolsCount,
			Error:      s.Error,
		})
	}
	writeJSON(w, http.StatusOK, mcpServersResponse{Servers: out})
}

// MCPExecute handles POST /v1/mcp/execute, running one tool by its namespaced
// name. A routing, gate, or transport failure comes back as an error result in
// the body rather than an HTTP error, so a caller always reads a tool outcome;
// only a missing subsystem or a malformed request is an HTTP-level failure.
func (d *Deps) MCPExecute(w http.ResponseWriter, r *http.Request) {
	if d.MCP == nil {
		writeError(w, http.StatusServiceUnavailable, "mcp_not_configured",
			"MCP not configured; start the server with -mcp-config")
		return
	}
	var req mcpExecuteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "malformed request body")
		return
	}
	if req.ToolName == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "tool_name is required")
		return
	}

	result, err := d.MCP.CallTool(r.Context(), req.ToolName, req.Arguments)
	if err != nil {
		writeJSON(w, http.StatusOK, mcpExecuteResponse{
			ToolName:     req.ToolName,
			IsError:      true,
			ErrorMessage: err.Error(),
		})
		return
	}
	writeJSON(w, http.StatusOK, mcpExecuteResponse{
		ToolName:     result.ToolName,
		Content:      result.Content,
		IsError:      result.IsError,
		ErrorMessage: result.ErrorMessage,
	})
}
