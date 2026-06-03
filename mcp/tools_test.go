// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"reflect"
	"testing"
)

func TestToolFullNameAndOpenAI(t *testing.T) {
	tool := Tool{ServerName: "files", Name: "read", Description: "read a file"}
	if tool.FullName() != "files__read" {
		t.Fatalf("full name=%q", tool.FullName())
	}
	def := tool.ToOpenAI()
	fn := def["function"].(map[string]any)
	if fn["name"] != "files__read" || fn["description"] != "read a file" {
		t.Fatalf("openai def wrong: %v", def)
	}
	// A nil schema becomes an empty object schema, never a missing field.
	params, ok := fn["parameters"].(map[string]any)
	if !ok || params["type"] != "object" {
		t.Fatalf("parameters should default to an object schema: %v", fn["parameters"])
	}
}

func TestToolResultToMessage(t *testing.T) {
	str := ToolResult{ToolName: "t", Content: "plain"}.ToMessage("call-1")
	if str["role"] != "tool" || str["tool_call_id"] != "call-1" || str["content"] != "plain" {
		t.Fatalf("string result wrong: %v", str)
	}

	obj := ToolResult{ToolName: "t", Content: map[string]any{"ok": true}}.ToMessage("c")
	if obj["content"] != `{"ok":true}` {
		t.Fatalf("object content should be JSON-encoded, got %v", obj["content"])
	}

	errRes := ToolResult{ToolName: "t", IsError: true, ErrorMessage: "boom"}.ToMessage("c")
	if errRes["content"] != "Error: boom" {
		t.Fatalf("error content wrong: %v", errRes["content"])
	}
}

func TestParseOpenAICall(t *testing.T) {
	server, tool, args := ParseOpenAICall(map[string]any{
		"function": map[string]any{
			"name":      "files__read",
			"arguments": `{"path":"a.txt","n":3}`,
		},
	})
	if server != "files" || tool != "read" {
		t.Fatalf("split wrong: server=%q tool=%q", server, tool)
	}
	if args["path"] != "a.txt" {
		t.Fatalf("args not decoded: %v", args)
	}
}

func TestParseOpenAICallObjectArguments(t *testing.T) {
	_, _, args := ParseOpenAICall(map[string]any{
		"function": map[string]any{
			"name":      "s__t",
			"arguments": map[string]any{"k": "v"},
		},
	})
	if args["k"] != "v" {
		t.Fatalf("object arguments should pass through: %v", args)
	}
}

func TestParseOpenAICallTolerantOfBadJSON(t *testing.T) {
	_, tool, args := ParseOpenAICall(map[string]any{
		"function": map[string]any{"name": "bare", "arguments": "{not json"},
	})
	if tool != "bare" {
		t.Fatalf("unnamespaced name should pass through as the tool: %q", tool)
	}
	if len(args) != 0 {
		t.Fatalf("malformed arguments should decode to empty, got %v", args)
	}
}

func TestMergeToolsUserWins(t *testing.T) {
	mcpTools := []Tool{
		{ServerName: "s", Name: "search", Description: "mcp search"},
		{ServerName: "s", Name: "read", Description: "mcp read"},
	}
	userOverride := map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "s__search", "description": "user search"},
	}
	userOnly := map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "calc", "description": "user calc"},
	}

	merged := MergeTools(mcpTools, []map[string]any{userOverride, userOnly})
	if len(merged) != 3 {
		t.Fatalf("want 3 merged tools, got %d", len(merged))
	}
	// The override replaced the MCP search in place.
	for _, m := range merged {
		fn := m["function"].(map[string]any)
		if fn["name"] == "s__search" && fn["description"] != "user search" {
			t.Fatalf("user tool should win on conflict: %v", fn)
		}
	}
}

func TestExtractAndHasToolCalls(t *testing.T) {
	resp := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"tool_calls": []any{
						map[string]any{"id": "1", "function": map[string]any{"name": "a__b"}},
					},
				},
			},
		},
	}
	if !HasToolCalls(resp) {
		t.Fatal("response with a tool call should report true")
	}
	calls := ExtractToolCalls(resp)
	if len(calls) != 1 || calls[0]["id"] != "1" {
		t.Fatalf("extracted calls wrong: %v", calls)
	}

	empty := map[string]any{"choices": []any{map[string]any{"message": map[string]any{}}}}
	if HasToolCalls(empty) {
		t.Fatal("a response with no tool calls should report false")
	}
	if got := ExtractToolCalls(map[string]any{}); got != nil {
		t.Fatalf("no choices should yield nil, got %v", got)
	}
}

func TestToolsToOpenAIRoundsTripSchema(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}}
	defs := ToolsToOpenAI([]Tool{{ServerName: "s", Name: "t", InputSchema: schema}})
	if len(defs) != 1 {
		t.Fatalf("want 1 def, got %d", len(defs))
	}
	got := defs[0]["function"].(map[string]any)["parameters"]
	if !reflect.DeepEqual(got, schema) {
		t.Fatalf("schema not preserved: %v", got)
	}
}
