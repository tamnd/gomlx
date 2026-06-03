// SPDX-License-Identifier: Apache-2.0

package mcp

import "encoding/json"

// A tool discovered on an MCP server has to travel through the OpenAI function
// calling format the model speaks and back again. This file holds the tool and
// result types and the conversions between the two formats. Tool names are
// namespaced so that two servers can each offer a "search" without colliding: the
// name on the wire is "server__tool", split back apart when a call comes in.

// NamespaceSeparator joins a server name and a tool name into the namespaced name
// the model sees.
const NamespaceSeparator = "__"

// Tool is a normalized tool offered by an MCP server.
type Tool struct {
	ServerName  string
	Name        string
	Description string
	InputSchema map[string]any
}

// FullName is the namespaced "server__tool" name used in OpenAI tool definitions.
func (t Tool) FullName() string {
	return t.ServerName + NamespaceSeparator + t.Name
}

// ToOpenAI renders the tool as an OpenAI function definition. A tool with no
// input schema is given an empty object schema, since the field is required.
func (t Tool) ToOpenAI() map[string]any {
	schema := t.InputSchema
	if schema == nil {
		schema = map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        t.FullName(),
			"description": t.Description,
			"parameters":  schema,
		},
	}
}

// ToolResult is the outcome of running a tool.
type ToolResult struct {
	ToolName     string
	Content      any
	IsError      bool
	ErrorMessage string
}

// ToMessage renders the result as an OpenAI "tool" role message answering the
// given tool call. An error becomes an "Error: ..." string, a string content is
// passed through, and any other content is JSON-encoded.
func (r ToolResult) ToMessage(toolCallID string) map[string]any {
	var content string
	switch {
	case r.IsError:
		content = "Error: " + r.ErrorMessage
	default:
		if s, ok := r.Content.(string); ok {
			content = s
		} else {
			raw, err := json.Marshal(r.Content)
			if err != nil {
				content = "Error: " + err.Error()
			} else {
				content = string(raw)
			}
		}
	}
	return map[string]any{
		"role":         "tool",
		"tool_call_id": toolCallID,
		"content":      content,
	}
}

// ToolsToOpenAI converts a list of tools to OpenAI function definitions.
func ToolsToOpenAI(tools []Tool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.ToOpenAI())
	}
	return out
}

// ParseOpenAICall splits an OpenAI tool call into its server name, tool name, and
// decoded arguments. The arguments field may arrive as a JSON string or as an
// already-decoded object; malformed JSON yields empty arguments rather than an
// error, matching how a model's stray output is tolerated elsewhere. A name with
// no namespace separator returns an empty server name for the caller to resolve.
func ParseOpenAICall(toolCall map[string]any) (server, tool string, args map[string]any) {
	fn, _ := toolCall["function"].(map[string]any)
	fullName, _ := fn["name"].(string)

	args = map[string]any{}
	switch a := fn["arguments"].(type) {
	case string:
		if a != "" {
			_ = json.Unmarshal([]byte(a), &args)
		}
	case map[string]any:
		args = a
	}

	if i := indexSep(fullName); i >= 0 {
		return fullName[:i], fullName[i+len(NamespaceSeparator):], args
	}
	return "", fullName, args
}

// indexSep returns the index of the first namespace separator in s, or -1.
func indexSep(s string) int {
	for i := 0; i+len(NamespaceSeparator) <= len(s); i++ {
		if s[i:i+len(NamespaceSeparator)] == NamespaceSeparator {
			return i
		}
	}
	return -1
}

// MergeTools combines MCP tools with user-provided OpenAI tools. User tools win on
// a name conflict, since a caller's explicit definition should override a
// discovered one. Order follows the MCP tools first, then any user-only tools.
func MergeTools(mcpTools []Tool, userTools []map[string]any) []map[string]any {
	merged := make([]map[string]any, 0, len(mcpTools)+len(userTools))
	index := map[string]int{}

	for _, t := range mcpTools {
		index[t.FullName()] = len(merged)
		merged = append(merged, t.ToOpenAI())
	}
	for _, ut := range userTools {
		fn, _ := ut["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if name == "" {
			merged = append(merged, ut)
			continue
		}
		if pos, ok := index[name]; ok {
			merged[pos] = ut
		} else {
			index[name] = len(merged)
			merged = append(merged, ut)
		}
	}
	return merged
}

// ExtractToolCalls returns the tool calls from an OpenAI-format model response.
func ExtractToolCalls(response map[string]any) []map[string]any {
	choices, _ := response["choices"].([]any)
	if len(choices) == 0 {
		return nil
	}
	first, _ := choices[0].(map[string]any)
	message, _ := first["message"].(map[string]any)
	raw, _ := message["tool_calls"].([]any)
	if len(raw) == 0 {
		return nil
	}
	calls := make([]map[string]any, 0, len(raw))
	for _, c := range raw {
		if call, ok := c.(map[string]any); ok {
			calls = append(calls, call)
		}
	}
	return calls
}

// HasToolCalls reports whether a response carries any tool calls.
func HasToolCalls(response map[string]any) bool {
	return len(ExtractToolCalls(response)) > 0
}
