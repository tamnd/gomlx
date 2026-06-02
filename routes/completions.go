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

var cmplIDSeq = newIDGen("cmpl")

// Completions handles POST /v1/completions (stream + non-stream).
func (d *Deps) Completions(w http.ResponseWriter, r *http.Request) {
	var req api.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "malformed JSON body: "+err.Error())
		return
	}
	prompt, ok := firstPrompt(req.Prompt)
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "prompt must be a string or array of strings")
		return
	}
	p := d.resolveSampling(req.MaxTokens, req.Temperature, req.TopP, req.MinP,
		req.RepetitionPenalty, req.PresencePenalty, req.FrequencyPenalty, req.TopK, req.Stop)
	model := req.Model
	if model == "" {
		model = d.Model
	}

	if req.Stream {
		d.streamCompletion(w, r, model, prompt, p)
		return
	}

	out, err := d.Engine.Generate(r.Context(), prompt, p)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	finish := out.FinishReason
	writeJSON(w, http.StatusOK, api.CompletionResponse{
		ID:      cmplIDSeq.next(),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []api.CompletionChoice{{Index: 0, Text: out.Text, FinishReason: &finish}},
		Usage: api.Usage{
			PromptTokens:     out.PromptTokens,
			CompletionTokens: out.CompletionTokens,
			TotalTokens:      out.PromptTokens + out.CompletionTokens,
		},
	})
}

// streamCompletion emits text_completion.chunk SSE events. The completion
// chunk shape is simpler than chat, so it is encoded directly here.
func (d *Deps) streamCompletion(w http.ResponseWriter, r *http.Request, model, prompt string, p engine.SamplingParams) {
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

	ch, err := d.Engine.StreamGenerate(r.Context(), prompt, p)
	if err != nil {
		return
	}
	bw := bufio.NewWriter(w)
	id := cmplIDSeq.next()
	created := time.Now().Unix()
	finish := "stop"
	for out := range ch {
		if out.Finished {
			finish = out.FinishReason
			continue
		}
		if err := writeCompletionChunk(bw, id, model, created, out.NewText, nil); err != nil {
			return
		}
	}
	_ = writeCompletionChunk(bw, id, model, created, "", &finish)
	_, _ = bw.WriteString("data: [DONE]\n\n")
	_ = bw.Flush()
}

// writeCompletionChunk writes one text_completion.chunk SSE event.
func writeCompletionChunk(bw *bufio.Writer, id, model string, created int64, text string, finish *string) error {
	resp := struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index        int     `json:"index"`
			Text         string  `json:"text"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}{ID: id, Object: "text_completion", Created: created, Model: model}
	resp.Choices = append(resp.Choices, struct {
		Index        int     `json:"index"`
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	}{Index: 0, Text: text, FinishReason: finish})
	b, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	if _, err := bw.WriteString("data: "); err != nil {
		return err
	}
	if _, err := bw.Write(b); err != nil {
		return err
	}
	if _, err := bw.WriteString("\n\n"); err != nil {
		return err
	}
	return bw.Flush()
}

// firstPrompt extracts the first prompt string from a string or []string body.
func firstPrompt(v any) (string, bool) {
	switch p := v.(type) {
	case string:
		return p, true
	case []any:
		if len(p) == 0 {
			return "", true
		}
		s, ok := p[0].(string)
		return s, ok
	default:
		return "", false
	}
}
