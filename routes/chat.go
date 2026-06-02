// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"bufio"
	"encoding/json"
	"net/http"
	"time"

	"github.com/tamnd/gomlx/api"
	"github.com/tamnd/gomlx/engine"
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
		d.streamChat(w, r, model, msgs, p, req.StreamOptions)
		return
	}

	out, err := d.Engine.Chat(r.Context(), msgs, p, toolDefs(req.Tools))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}

	content := out.Text
	finish := out.FinishReason
	resp := api.ChatCompletionResponse{
		ID:      chatIDSeq.next(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []api.ChatCompletionChoice{{
			Index:        0,
			Message:      api.AssistantMessage{Role: "assistant", Content: &content},
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

// streamChat drives an SSE response from the engine's streaming channel.
func (d *Deps) streamChat(w http.ResponseWriter, r *http.Request, model string, msgs []engine.ChatMessage, p engine.SamplingParams, so *api.StreamOptions) {
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

	ch, err := d.Engine.StreamChat(r.Context(), msgs, p, nil)
	if err != nil {
		// Headers already sent; emit an error chunk best-effort.
		return
	}

	bw := bufio.NewWriter(w)
	enc := api.NewStreamEncoder(bw, chatIDSeq.next(), model, time.Now().Unix())
	_ = enc.Role()

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
		if err := enc.Content(out.NewText, reasoning); err != nil {
			return // client disconnected
		}
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
