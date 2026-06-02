// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"bufio"
	"encoding/json"
	"net/http"
	"time"

	"github.com/tamnd/gomlx/api"
	"github.com/tamnd/gomlx/engine"
	"github.com/tamnd/gomlx/toolparsers"
)

// idCounter feeds deterministic-ish response IDs without a clock dependency in
// the hot path; uniqueness within a process is enough for clients.
var chatIDSeq = newIDGen("chatcmpl")

// ChatCompletions handles POST /v1/chat/completions (stream + non-stream).
func (d *Deps) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req api.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "malformed JSON body: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "messages must not be empty")
		return
	}

	// max_completion_tokens supersedes max_tokens (newer OpenAI SDKs).
	maxTok := req.MaxTokens
	if req.MaxCompletionTokens != nil {
		maxTok = req.MaxCompletionTokens
	}
	p := d.resolveSampling(maxTok, req.Temperature, req.TopP, req.MinP,
		req.RepetitionPenalty, req.PresencePenalty, req.FrequencyPenalty, req.TopK, req.Stop)

	msgs := toChatMessages(req.Messages)
	model := req.Model
	if model == "" {
		model = d.Model
	}

	if req.Stream {
		d.streamChat(w, r, model, msgs, p, &req, req.StreamOptions)
		return
	}

	out, err := d.Engine.Chat(r.Context(), msgs, p, toolDefs(req.Tools))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	content := out.Text
	msg := api.AssistantMessage{Role: "assistant", Content: &content}
	finish := out.FinishReason

	// When the client offers tools, run the configured parser over the model
	// output. A successful extraction replaces the message content with the
	// leftover text (nil when the call consumed the whole reply) and surfaces
	// the calls under tool_calls with a tool_calls finish reason.
	if len(req.Tools) > 0 {
		if parser, ok := toolparsers.Get(d.parserName()); ok {
			res := parser.ExtractToolCalls(out.Text, toolParserRequest(&req))
			if res.ToolsCalled {
				msg.Content = res.Content
				msg.ToolCalls = toAPIToolCalls(res.ToolCalls)
				finish = "tool_calls"
			}
		}
	}

	resp := api.ChatCompletionResponse{
		ID:      chatIDSeq.next(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []api.ChatCompletionChoice{{
			Index:        0,
			Message:      msg,
			FinishReason: &finish,
		}},
		Usage: api.Usage{
			PromptTokens:     out.PromptTokens,
			CompletionTokens: out.CompletionTokens,
			TotalTokens:      out.PromptTokens + out.CompletionTokens,
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

// parserName returns the configured tool-call parser, defaulting to the auto
// parser when none was set.
func (d *Deps) parserName() string {
	if d.ToolCallParser == "" {
		return "auto"
	}
	return d.ToolCallParser
}

// toolParserRequest projects the chat request into the loosely typed view the
// tool parsers consult for argument type coercion. Only the tool definitions
// matter, so an absence of tools yields a nil request.
func toolParserRequest(req *api.ChatCompletionRequest) toolparsers.Request {
	if len(req.Tools) == 0 {
		return nil
	}
	tools := make([]any, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = map[string]any{"type": t.Type, "function": t.Function}
	}
	return toolparsers.Request{"tools": tools}
}

// toAPIToolCalls converts parser tool calls to the OpenAI response shape.
func toAPIToolCalls(calls []toolparsers.ToolCall) []api.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]api.ToolCall, len(calls))
	for i, c := range calls {
		out[i] = api.ToolCall{
			ID:       c.ID,
			Type:     "function",
			Function: api.FunctionCall{Name: c.Name, Arguments: c.Arguments},
		}
	}
	return out
}

// streamDeltaJSON is the minimal delta envelope used to emit a streaming
// tool-call chunk. The tool-call entries already carry the OpenAI field shape.
type streamDeltaJSON struct {
	ToolCalls []toolparsers.StreamToolCall `json:"tool_calls"`
}

// streamChat drives an SSE response from the engine's streaming channel. When
// the request offers tools, content deltas are fed through the streaming tool
// parser, which decides what to surface as content versus tool-call fragments.
func (d *Deps) streamChat(w http.ResponseWriter, r *http.Request, model string, msgs []engine.ChatMessage, p engine.SamplingParams, req *api.ChatCompletionRequest, so *api.StreamOptions) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	ch, err := d.Engine.StreamChat(r.Context(), msgs, p, toolDefs(req.Tools))
	if err != nil {
		// Headers already sent; emit an error chunk best-effort.
		return
	}

	bw := bufio.NewWriter(w)
	enc := api.NewStreamEncoder(bw, chatIDSeq.next(), model, time.Now().Unix())
	_ = enc.Role()

	// Set up the streaming tool parser when tools are on offer.
	var parser toolparsers.ToolParser
	var parserReq toolparsers.Request
	if len(req.Tools) > 0 {
		if pp, ok := toolparsers.Get(d.parserName()); ok {
			parser = pp
			parserReq = toolParserRequest(req)
		}
	}
	prevText := "" // running model output fed to the parser
	sawToolCall := false

	var usage *api.Usage
	finish := "stop"
	for out := range ch {
		if out.Finished {
			finish = out.FinishReason
			if so != nil && so.IncludeUsage {
				usage = &api.Usage{
					PromptTokens:     out.PromptTokens,
					CompletionTokens: out.CompletionTokens,
					TotalTokens:      out.PromptTokens + out.CompletionTokens,
				}
			}
			continue
		}
		reasoning := out.Channel == engine.ChannelReasoning

		// Reasoning is never tool markup; pass it straight through. Content is
		// routed through the parser when one is active.
		if parser == nil || reasoning {
			if err := enc.Content(out.NewText, reasoning); err != nil {
				return // client disconnected
			}
			continue
		}

		curText := prevText + out.NewText
		delta := parser.ExtractToolCallsStreaming(prevText, curText, out.NewText, parserReq)
		prevText = curText
		if delta == nil {
			continue
		}
		if delta.Content != nil {
			if err := enc.Content(*delta.Content, false); err != nil {
				return
			}
		}
		if len(delta.ToolCalls) > 0 {
			sawToolCall = true
			b, mErr := json.Marshal(streamDeltaJSON{ToolCalls: delta.ToolCalls})
			if mErr == nil {
				if err := enc.DeltaJSON(b); err != nil {
					return
				}
			}
		}
	}
	if sawToolCall && finish == "stop" {
		finish = "tool_calls"
	}
	_ = enc.Finish(finish, usage)
	_ = enc.Done()
}

// toolDefs converts API tool definitions to the engine's opaque tool slice.
func toolDefs(tools []api.ToolDefinition) []any {
	if len(tools) == 0 {
		return nil
	}
	out := make([]any, len(tools))
	for i, t := range tools {
		out[i] = t
	}
	return out
}
