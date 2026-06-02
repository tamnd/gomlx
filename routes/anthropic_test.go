// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/gomlx/api"
)

func anthropicToolReq(stream bool) api.AnthropicRequest {
	return api.AnthropicRequest{
		Model:     "test",
		MaxTokens: 64,
		Stream:    stream,
		Messages: []api.AnthropicMessage{{
			Role:    "user",
			Content: json.RawMessage(`"weather in Paris?"`),
		}},
		Tools: []api.AnthropicToolDef{{
			Name:        "get_weather",
			Description: "look up weather",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
		}},
	}
}

func TestAnthropicMessagesToolUse(t *testing.T) {
	d := newCannedDeps(hermesWeather, nil, "hermes")
	body, _ := json.Marshal(anthropicToolReq(false))
	rr := httptest.NewRecorder()
	d.AnthropicMessages(rr, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body))))

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got %d body=%s", rr.Code, rr.Body.String())
	}
	var resp api.AnthropicResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Type != "message" || resp.Role != "assistant" {
		t.Errorf("envelope: type=%q role=%q", resp.Type, resp.Role)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop_reason: got %q want tool_use", resp.StopReason)
	}
	var toolBlocks int
	for _, b := range resp.Content {
		if b.Type == "tool_use" {
			toolBlocks++
			if b.Name != "get_weather" {
				t.Errorf("tool name: got %q", b.Name)
			}
			if string(b.Input) != `{"city":"Paris"}` {
				t.Errorf("tool input: got %s", string(b.Input))
			}
		}
	}
	if toolBlocks != 1 {
		t.Fatalf("tool_use blocks: got %d want 1", toolBlocks)
	}
}

func TestAnthropicMessagesText(t *testing.T) {
	d := newCannedDeps("Hello there.", nil, "hermes")
	req := api.AnthropicRequest{
		Model:     "test",
		MaxTokens: 64,
		Messages:  []api.AnthropicMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	body, _ := json.Marshal(req)
	rr := httptest.NewRecorder()
	d.AnthropicMessages(rr, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body))))

	var resp api.AnthropicResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop_reason: got %q want end_turn", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" {
		t.Fatalf("content: %+v", resp.Content)
	}
	if resp.Content[0].Text == nil || *resp.Content[0].Text != "Hello there." {
		t.Errorf("text block: got %v", resp.Content[0].Text)
	}
}

func TestAnthropicMessagesMissingMaxTokens(t *testing.T) {
	d := newCannedDeps("hi", nil, "")
	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	rr := httptest.NewRecorder()
	d.AnthropicMessages(rr, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want 400", rr.Code)
	}
}

func TestAnthropicMessagesStreamingToolUse(t *testing.T) {
	deltas := []string{`<tool_call>`, `{"name": "get_weather", "arguments": {"city": "Paris"}}`, `</tool_call>`}
	d := newCannedDeps("", deltas, "hermes")
	body, _ := json.Marshal(anthropicToolReq(true))
	rr := httptest.NewRecorder()
	d.AnthropicMessages(rr, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body))))

	if ct := rr.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type: got %q", ct)
	}
	out := rr.Body.String()

	// Parse the SSE event sequence.
	var events []string
	var sawMessageStart, sawMessageStop bool
	var toolName, toolInput, stopReason string
	for _, block := range strings.Split(out, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var ev, data string
		for _, line := range strings.Split(block, "\n") {
			if strings.HasPrefix(line, "event: ") {
				ev = strings.TrimPrefix(line, "event: ")
			}
			if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
			}
		}
		events = append(events, ev)
		var payload map[string]any
		_ = json.Unmarshal([]byte(data), &payload)
		switch ev {
		case "message_start":
			sawMessageStart = true
		case "message_stop":
			sawMessageStop = true
		case "content_block_start":
			if cb, ok := payload["content_block"].(map[string]any); ok && cb["type"] == "tool_use" {
				toolName, _ = cb["name"].(string)
			}
		case "content_block_delta":
			if dl, ok := payload["delta"].(map[string]any); ok && dl["type"] == "input_json_delta" {
				toolInput, _ = dl["partial_json"].(string)
			}
		case "message_delta":
			if dl, ok := payload["delta"].(map[string]any); ok {
				stopReason, _ = dl["stop_reason"].(string)
			}
		}
	}

	if !sawMessageStart || !sawMessageStop {
		t.Fatalf("missing message_start/stop in: %v", events)
	}
	if events[0] != "message_start" {
		t.Errorf("first event: got %q", events[0])
	}
	if toolName != "get_weather" {
		t.Errorf("tool name: got %q", toolName)
	}
	if toolInput != `{"city": "Paris"}` {
		t.Errorf("tool input: got %q", toolInput)
	}
	if stopReason != "tool_use" {
		t.Errorf("stop_reason: got %q want tool_use", stopReason)
	}
}
