// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/gomlx/api"
	"github.com/tamnd/gomlx/engine"
)

// cannedEngine returns a fixed reply, so a route test can drive a known
// tool-call wire format through the parser. Streaming replays the supplied
// deltas. It embeds MockEngine for the interface methods it does not override.
type cannedEngine struct {
	*engine.MockEngine
	full   string
	deltas []string
}

func (c *cannedEngine) Chat(_ context.Context, _ []engine.ChatMessage, _ engine.SamplingParams, _ []any) (engine.GenerationOutput, error) {
	return engine.GenerationOutput{
		Text:             c.full,
		PromptTokens:     3,
		CompletionTokens: 5,
		FinishReason:     "stop",
		Finished:         true,
		Channel:          engine.ChannelContent,
	}, nil
}

func (c *cannedEngine) StreamChat(ctx context.Context, _ []engine.ChatMessage, _ engine.SamplingParams, _ []any) (<-chan engine.GenerationOutput, error) {
	ch := make(chan engine.GenerationOutput)
	go func() {
		defer close(ch)
		for _, d := range c.deltas {
			select {
			case <-ctx.Done():
				return
			case ch <- engine.GenerationOutput{NewText: d, Channel: engine.ChannelContent}:
			}
		}
		ch <- engine.GenerationOutput{Finished: true, FinishReason: "stop", PromptTokens: 3, CompletionTokens: 5, Channel: engine.ChannelContent}
	}()
	return ch, nil
}

func newCannedDeps(full string, deltas []string, parser string) *Deps {
	return &Deps{
		Engine:           &cannedEngine{MockEngine: engine.NewMockEngine("test"), full: full, deltas: deltas},
		Model:            "test",
		DefaultMaxTokens: 64,
		ToolCallParser:   parser,
	}
}

const hermesWeather = `<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>`

func weatherToolReq(stream bool) api.ChatCompletionRequest {
	return api.ChatCompletionRequest{
		Model:    "test",
		Messages: []api.Message{{Role: "user", Content: "weather in Paris?"}},
		Stream:   stream,
		Tools: []api.ToolDefinition{{
			Type: "function",
			Function: map[string]any{
				"name": "get_weather",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
				},
			},
		}},
	}
}

func TestChatCompletionsToolCall(t *testing.T) {
	d := newCannedDeps(hermesWeather, nil, "hermes")
	body, _ := json.Marshal(weatherToolReq(false))
	rr := httptest.NewRecorder()
	d.ChatCompletions(rr, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body))))

	if rr.Code != http.StatusOK {
		t.Fatalf("status: got %d body=%s", rr.Code, rr.Body.String())
	}
	var resp api.ChatCompletionResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	choice := resp.Choices[0]
	if choice.FinishReason == nil || *choice.FinishReason != "tool_calls" {
		t.Errorf("finish_reason: got %v want tool_calls", choice.FinishReason)
	}
	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("tool_calls: got %d want 1", len(choice.Message.ToolCalls))
	}
	tc := choice.Message.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("name: got %q", tc.Function.Name)
	}
	if tc.Function.Arguments != `{"city": "Paris"}` {
		t.Errorf("arguments: got %q", tc.Function.Arguments)
	}
	if tc.Type != "function" {
		t.Errorf("type: got %q", tc.Type)
	}
	// Content is consumed entirely by the call.
	if choice.Message.Content != nil && *choice.Message.Content != "" {
		t.Errorf("content: got %q want empty/null", *choice.Message.Content)
	}
}

// TestChatCompletionsNoToolsPassthrough confirms the parser does not run when
// the request offers no tools: the raw text becomes the content verbatim.
func TestChatCompletionsNoToolsPassthrough(t *testing.T) {
	d := newCannedDeps(hermesWeather, nil, "hermes")
	req := api.ChatCompletionRequest{
		Model:    "test",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	}
	body, _ := json.Marshal(req)
	rr := httptest.NewRecorder()
	d.ChatCompletions(rr, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body))))

	var resp api.ChatCompletionResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	choice := resp.Choices[0]
	if len(choice.Message.ToolCalls) != 0 {
		t.Errorf("tool_calls: got %d want 0", len(choice.Message.ToolCalls))
	}
	if choice.Message.Content == nil || *choice.Message.Content != hermesWeather {
		t.Errorf("content not passed through verbatim: got %v", choice.Message.Content)
	}
	if choice.FinishReason == nil || *choice.FinishReason != "stop" {
		t.Errorf("finish_reason: got %v want stop", choice.FinishReason)
	}
}

func TestChatCompletionsStreamingToolCall(t *testing.T) {
	deltas := []string{`<tool_call>`, `{"name": "get_weather", "arguments": {"city": "Paris"}}`, `</tool_call>`}
	d := newCannedDeps("", deltas, "hermes")
	body, _ := json.Marshal(weatherToolReq(true))
	rr := httptest.NewRecorder()
	d.ChatCompletions(rr, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body))))

	if ct := rr.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type: got %q", ct)
	}
	out := rr.Body.String()
	if !strings.Contains(out, "[DONE]") {
		t.Fatalf("missing DONE sentinel:\n%s", out)
	}

	// Collect the tool-call fragments emitted across chunks.
	var gotName, gotArgs string
	var finish string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimPrefix(line, "data: ")
		line = strings.TrimSpace(line)
		if line == "" || line == "[DONE]" {
			continue
		}
		var chunk api.ChatCompletionChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}
		ch := chunk.Choices[0]
		for _, tc := range ch.Delta.ToolCalls {
			gotName += tc.Function.Name
			gotArgs += tc.Function.Arguments
		}
		if ch.FinishReason != nil {
			finish = *ch.FinishReason
		}
	}
	if gotName != "get_weather" {
		t.Errorf("streamed name: got %q", gotName)
	}
	if gotArgs != `{"city": "Paris"}` {
		t.Errorf("streamed arguments: got %q", gotArgs)
	}
	if finish != "tool_calls" {
		t.Errorf("finish_reason: got %q want tool_calls", finish)
	}
}
