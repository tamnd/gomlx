// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/tamnd/gomlx/api"
	"github.com/tamnd/gomlx/engine"
	"github.com/tamnd/gomlx/toolparsers"
)

var anthropicIDSeq = newIDGen("msg")

// AnthropicMessages handles POST /v1/messages. It translates the Anthropic
// request into the OpenAI-shaped engine path, runs inference, parses tool
// calls, and renders the reply back into Anthropic content blocks.
func (d *Deps) AnthropicMessages(w http.ResponseWriter, r *http.Request) {
	var req api.AnthropicRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "malformed JSON body: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "messages must not be empty")
		return
	}
	if req.MaxTokens <= 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "max_tokens must be a positive integer")
		return
	}

	chatReq := anthropicToChatRequest(&req)
	p := d.resolveSampling(chatReq.MaxTokens, chatReq.Temperature, chatReq.TopP, nil,
		nil, nil, nil, chatReq.TopK, chatReq.Stop)
	msgs := toChatMessages(chatReq.Messages)
	model := req.Model
	if model == "" {
		model = d.Model
	}

	if req.Stream {
		d.streamAnthropic(w, r, model, msgs, p, &chatReq)
		return
	}

	out, err := d.Engine.Chat(r.Context(), msgs, p, toolDefs(chatReq.Tools))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	contentText := out.Text
	var toolCalls []toolparsers.ToolCall
	finish := out.FinishReason
	if len(chatReq.Tools) > 0 {
		if parser, ok := toolparsers.Get(d.parserName()); ok {
			res := parser.ExtractToolCalls(out.Text, toolParserRequest(&chatReq))
			if res.ToolsCalled {
				contentText = ""
				if res.Content != nil {
					contentText = *res.Content
				}
				toolCalls = res.ToolCalls
				finish = "tool_calls"
			}
		}
	}

	var blocks []api.AnthropicResponseBlock
	if out.ReasoningText != "" {
		blocks = append(blocks, api.AnthropicResponseBlock{Type: "thinking", Thinking: strPtr(out.ReasoningText)})
	}
	if contentText != "" {
		blocks = append(blocks, api.AnthropicResponseBlock{Type: "text", Text: strPtr(contentText)})
	}
	for _, tc := range toolCalls {
		blocks = append(blocks, api.AnthropicResponseBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Name,
			Input: toolInputJSON(tc.Arguments),
		})
	}
	// Anthropic always expects at least one content block.
	if len(blocks) == 0 {
		blocks = append(blocks, api.AnthropicResponseBlock{Type: "text", Text: strPtr("")})
	}

	resp := api.AnthropicResponse{
		ID:         anthropicIDSeq.next(),
		Type:       "message",
		Role:       "assistant",
		Model:      model,
		Content:    blocks,
		StopReason: anthropicStopReason(finish),
		Usage: api.AnthropicUsage{
			InputTokens:  out.PromptTokens,
			OutputTokens: out.CompletionTokens,
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

// anthropicToChatRequest mirrors the reference anthropic_to_openai adapter:
// system becomes a leading system message, content blocks expand to OpenAI
// messages (tool_use to assistant tool_calls, tool_result to tool messages),
// and tools take the OpenAI function shape.
func anthropicToChatRequest(req *api.AnthropicRequest) api.ChatCompletionRequest {
	var msgs []api.Message
	if sys := req.SystemText(); sys != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: sys})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, convertAnthropicMessage(m)...)
	}

	var tools []api.ToolDefinition
	for _, t := range req.Tools {
		params := t.InputSchema
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		tools = append(tools, api.ToolDefinition{
			Type:     "function",
			Function: map[string]any{"name": t.Name, "description": t.Description, "parameters": params},
		})
	}

	maxTok := req.MaxTokens
	return api.ChatCompletionRequest{
		Model:       req.Model,
		Messages:    msgs,
		MaxTokens:   &maxTok,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		Stream:      req.Stream,
		Stop:        req.StopSequences,
		Tools:       tools,
	}
}

// convertAnthropicMessage expands one Anthropic message into OpenAI messages.
func convertAnthropicMessage(m api.AnthropicMessage) []api.Message {
	blocks := m.Blocks()
	if len(blocks) == 1 && blocks[0].Type == "text" && blocks[0].ID == "" {
		return []api.Message{{Role: m.Role, Content: blocks[0].Text}}
	}

	var textParts []string
	var toolCalls []api.ToolCall
	var toolResults []api.Message
	for _, b := range blocks {
		switch b.Type {
		case "text":
			textParts = append(textParts, b.Text)
		case "tool_use":
			id := b.ID
			if id == "" {
				id = "call_" + anthropicIDSeq.next()
			}
			args := "{}"
			if b.Input != nil {
				if enc, err := json.Marshal(b.Input); err == nil {
					args = string(enc)
				}
			}
			toolCalls = append(toolCalls, api.ToolCall{
				ID:       id,
				Type:     "function",
				Function: api.FunctionCall{Name: b.Name, Arguments: args},
			})
		case "tool_result":
			toolResults = append(toolResults, api.Message{
				Role:       "tool",
				Content:    toolResultText(b.Content),
				ToolCallID: b.ToolUseID,
			})
		}
	}

	combined := joinNonEmpty(textParts, "\n")
	var out []api.Message
	switch m.Role {
	case "assistant":
		if len(toolCalls) > 0 {
			out = append(out, api.Message{Role: "assistant", Content: combined, ToolCalls: toolCalls})
		} else {
			out = append(out, api.Message{Role: "assistant", Content: combined})
		}
	case "user":
		if len(textParts) > 0 {
			out = append(out, api.Message{Role: "user", Content: combined})
		}
		out = append(out, toolResults...)
		if len(textParts) == 0 && len(toolResults) == 0 {
			out = append(out, api.Message{Role: "user", Content: ""})
		}
	default:
		out = append(out, api.Message{Role: m.Role, Content: combined})
	}
	return out
}

// toolResultText flattens an Anthropic tool_result content payload (string or
// list of text blocks) to plain text.
func toolResultText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var items []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return ""
	}
	var parts []string
	for _, it := range items {
		if it.Text != "" {
			parts = append(parts, it.Text)
		}
	}
	return joinNonEmpty(parts, "\n")
}

// streamAnthropic emits the Anthropic SSE event sequence: message_start, then
// content blocks (thinking/text) as the engine streams, tool_use blocks once
// the calls are complete, message_delta with the stop reason, and message_stop.
func (d *Deps) streamAnthropic(w http.ResponseWriter, r *http.Request, model string, msgs []engine.ChatMessage, p engine.SamplingParams, req *api.ChatCompletionRequest) {
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

	msgID := anthropicIDSeq.next()
	ctx, stop := d.trackStream(r.Context(), msgID)
	defer stop()

	ch, err := d.Engine.StreamChat(ctx, msgs, p, toolDefs(req.Tools))
	if err != nil {
		return
	}

	bw := bufio.NewWriter(w)
	emit := func(event string, data any) bool {
		b, mErr := json.Marshal(data)
		if mErr != nil {
			return true
		}
		if _, wErr := fmt.Fprintf(bw, "event: %s\ndata: %s\n\n", event, b); wErr != nil {
			return false
		}
		return bw.Flush() == nil
	}

	emit("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": msgID, "type": "message", "role": "assistant", "model": model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	})

	var parser toolparsers.ToolParser
	var parserReq toolparsers.Request
	if len(req.Tools) > 0 {
		if pp, ok := toolparsers.Get(d.parserName()); ok {
			parser = pp
			parserReq = toolParserRequest(req)
		}
	}

	bs := &blockStreamer{emit: emit}
	tools := newToolAccum()
	prevText := ""
	promptTokens, completionTokens := 0, 0
	finish := "stop"

	for out := range ch {
		if out.Finished {
			finish = out.FinishReason
			promptTokens = out.PromptTokens
			completionTokens = out.CompletionTokens
			continue
		}
		if out.Channel == engine.ChannelReasoning {
			if !bs.delta("thinking", out.NewText) {
				return
			}
			continue
		}
		// Content channel.
		if parser == nil {
			if !bs.delta("text", out.NewText) {
				return
			}
			continue
		}
		curText := prevText + out.NewText
		delta := parser.ExtractToolCallsStreaming(prevText, curText, out.NewText, parserReq)
		prevText = curText
		if delta == nil {
			continue
		}
		if delta.Content != nil && *delta.Content != "" {
			if !bs.delta("text", *delta.Content) {
				return
			}
		}
		tools.add(delta.ToolCalls)
	}

	bs.close()

	calls := tools.finalize()
	for _, tc := range calls {
		idx := bs.nextIndex()
		ok1 := emit("content_block_start", map[string]any{
			"type": "content_block_start", "index": idx,
			"content_block": map[string]any{"type": "tool_use", "id": tc.ID, "name": tc.Name, "input": map[string]any{}},
		})
		ok2 := emit("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": idx,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": tc.Arguments},
		})
		ok3 := emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
		if !ok1 || !ok2 || !ok3 {
			return
		}
	}

	stopReason := anthropicStopReason(finish)
	if len(calls) > 0 {
		stopReason = "tool_use"
	}
	emit("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": map[string]any{"input_tokens": promptTokens, "output_tokens": completionTokens},
	})
	emit("message_stop", map[string]any{"type": "message_stop"})
}

// blockStreamer tracks the open content block and emits the start/delta/stop
// events as the active block type changes.
type blockStreamer struct {
	emit    func(string, any) bool
	current string // "", "text", or "thinking"
	index   int
	opened  bool
}

// delta emits a delta for blockType, opening or switching the active block as
// needed. It returns false when the client connection drops.
func (b *blockStreamer) delta(blockType, text string) bool {
	if text == "" {
		return true
	}
	if b.current != blockType {
		if b.opened {
			if !b.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": b.index}) {
				return false
			}
			b.index++
		}
		b.current = blockType
		b.opened = true
		start := map[string]any{"type": blockType, "text": ""}
		if blockType == "thinking" {
			start = map[string]any{"type": blockType, "thinking": ""}
		}
		if !b.emit("content_block_start", map[string]any{
			"type": "content_block_start", "index": b.index, "content_block": start,
		}) {
			return false
		}
	}
	deltaType, key := "text_delta", "text"
	if blockType == "thinking" {
		deltaType, key = "thinking_delta", "thinking"
	}
	return b.emit("content_block_delta", map[string]any{
		"type": "content_block_delta", "index": b.index,
		"delta": map[string]any{"type": deltaType, key: text},
	})
}

// close shuts the currently open block, if any.
func (b *blockStreamer) close() {
	if b.opened {
		b.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": b.index})
		b.index++
		b.opened = false
		b.current = ""
	}
}

// nextIndex returns the next free block index and advances the counter, for the
// tool_use blocks appended after the content blocks.
func (b *blockStreamer) nextIndex() int {
	i := b.index
	b.index++
	return i
}

// toolAccum reassembles streaming tool-call fragments (which may arrive split
// across deltas by index) into complete calls.
type toolAccum struct {
	order []int
	byIdx map[int]*toolBuild
}

type toolBuild struct {
	id, name string
	args     []byte
}

func newToolAccum() *toolAccum { return &toolAccum{byIdx: map[int]*toolBuild{}} }

func (a *toolAccum) add(entries []toolparsers.StreamToolCall) {
	for _, e := range entries {
		tb, ok := a.byIdx[e.Index]
		if !ok {
			tb = &toolBuild{}
			a.byIdx[e.Index] = tb
			a.order = append(a.order, e.Index)
		}
		if e.ID != nil && *e.ID != "" {
			tb.id = *e.ID
		}
		if e.Function != nil {
			if e.Function.Name != nil && *e.Function.Name != "" {
				tb.name = *e.Function.Name
			}
			if e.Function.Arguments != nil {
				tb.args = append(tb.args, *e.Function.Arguments...)
			}
		}
	}
}

func (a *toolAccum) finalize() []toolparsers.ToolCall {
	sort.Ints(a.order)
	var out []toolparsers.ToolCall
	for _, idx := range a.order {
		tb := a.byIdx[idx]
		if tb.name == "" {
			continue
		}
		args := string(tb.args)
		if args == "" {
			args = "{}"
		}
		id := tb.id
		if id == "" {
			id = "call_" + anthropicIDSeq.next()
		}
		out = append(out, toolparsers.ToolCall{ID: id, Name: tb.name, Arguments: args})
	}
	return out
}

// anthropicStopReason maps an OpenAI finish reason to the Anthropic vocabulary.
func anthropicStopReason(openaiReason string) string {
	switch openaiReason {
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	case "stop", "content_filter", "":
		return "end_turn"
	default:
		return "end_turn"
	}
}

// toolInputJSON re-encodes a tool-call argument string as a compact JSON object
// for the tool_use input field, defaulting to {} on bad input.
func toolInputJSON(arguments string) json.RawMessage {
	var m map[string]any
	if err := json.Unmarshal([]byte(arguments), &m); err != nil {
		return json.RawMessage("{}")
	}
	enc, err := json.Marshal(m)
	if err != nil {
		return json.RawMessage("{}")
	}
	return enc
}

// strPtr returns a pointer to s.
func strPtr(s string) *string { return &s }

// joinNonEmpty joins parts with sep.
func joinNonEmpty(parts []string, sep string) string {
	return strings.Join(parts, sep)
}
