// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/gomlx/mcp"
)

// configuredManager returns a manager whose registry already holds a couple of
// tools, without connecting to any real server. That is enough to exercise the
// listing handler and the not-connected execute path; the manager's own
// connection behaviour is covered in the mcp package.
func configuredManager(t *testing.T) *mcp.Manager {
	t.Helper()
	mgr := mcp.NewManager(mcp.Config{}, mcp.ClientInfo{Name: "gomlx"})
	mgr.Registry().SetTools("web", []mcp.Tool{
		{ServerName: "web", Name: "read", Description: "read a page", InputSchema: map[string]any{"type": "object"}},
		{ServerName: "web", Name: "write", Description: "write a page"},
	})
	return mgr
}

func TestMCPToolsUnconfigured(t *testing.T) {
	d := &Deps{}
	rec := httptest.NewRecorder()
	d.MCPTools(rec, httptest.NewRequest(http.MethodGet, "/v1/mcp/tools", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var out mcpToolsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Count != 0 || len(out.Tools) != 0 {
		t.Fatalf("unconfigured tools should be empty, got %+v", out)
	}
}

func TestMCPToolsListsRegistered(t *testing.T) {
	d := &Deps{MCP: configuredManager(t)}
	rec := httptest.NewRecorder()
	d.MCPTools(rec, httptest.NewRequest(http.MethodGet, "/v1/mcp/tools", nil))

	var out mcpToolsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Count != 2 {
		t.Fatalf("count=%d, want 2", out.Count)
	}
	first := out.Tools[0]
	if first.Name != "web__read" || first.Server != "web" || first.Description != "read a page" {
		t.Fatalf("first tool wrong: %+v", first)
	}
	if first.Parameters["type"] != "object" {
		t.Fatalf("parameters not carried through: %+v", first.Parameters)
	}
}

func TestMCPServersUnconfigured(t *testing.T) {
	d := &Deps{}
	rec := httptest.NewRecorder()
	d.MCPServers(rec, httptest.NewRequest(http.MethodGet, "/v1/mcp/servers", nil))

	var out mcpServersResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Servers) != 0 {
		t.Fatalf("unconfigured servers should be empty, got %+v", out)
	}
}

func TestMCPExecuteUnconfigured(t *testing.T) {
	d := &Deps{}
	rec := httptest.NewRecorder()
	body := strings.NewReader(`{"tool_name":"web__read","arguments":{}}`)
	d.MCPExecute(rec, httptest.NewRequest(http.MethodPost, "/v1/mcp/execute", body))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d, want 503", rec.Code)
	}
}

func TestMCPExecuteRequiresToolName(t *testing.T) {
	d := &Deps{MCP: configuredManager(t)}
	rec := httptest.NewRecorder()
	d.MCPExecute(rec, httptest.NewRequest(http.MethodPost, "/v1/mcp/execute", strings.NewReader(`{}`)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d, want 400", rec.Code)
	}
}

func TestMCPExecuteNotConnectedIsErrorInBody(t *testing.T) {
	// The tool is registered but its server was never connected, so the call
	// fails to route. That surfaces as an error result in the body, not an HTTP
	// error, so a client always reads a tool outcome.
	d := &Deps{MCP: configuredManager(t)}
	rec := httptest.NewRecorder()
	body := strings.NewReader(`{"tool_name":"web__read","arguments":{"url":"x"}}`)
	d.MCPExecute(rec, httptest.NewRequest(http.MethodPost, "/v1/mcp/execute", body))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d, want 200", rec.Code)
	}
	var out mcpExecuteResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if !out.IsError || out.ErrorMessage == "" {
		t.Fatalf("expected an error result, got %+v", out)
	}
}
