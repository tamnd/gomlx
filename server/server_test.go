// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/gomlx/api"
	"github.com/tamnd/gomlx/config"
	"github.com/tamnd/gomlx/engine"
)

func newTestServer(apiKey string) *httptest.Server {
	cfg := config.Default()
	cfg.Model = "qwen3.5-4b"
	cfg.APIKey = apiKey
	app := New(cfg, engine.NewMockEngine(cfg.Model), nil)
	return httptest.NewServer(app.Handler())
}

func TestChatCompletionNonStream(t *testing.T) {
	ts := newTestServer("")
	defer ts.Close()

	body := `{"model":"qwen3.5-4b","messages":[{"role":"user","content":"hello world"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: got %d", resp.StatusCode)
	}
	var out api.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "chat.completion" || len(out.Choices) != 1 {
		t.Fatalf("unexpected response: %+v", out)
	}
	if out.Choices[0].Message.Content == nil || *out.Choices[0].Message.Content == "" {
		t.Error("empty content")
	}
	if out.Choices[0].Message.Role != "assistant" {
		t.Errorf("role: got %q", out.Choices[0].Message.Role)
	}
	if out.Usage.TotalTokens != out.Usage.PromptTokens+out.Usage.CompletionTokens {
		t.Error("usage totals inconsistent")
	}
}

func TestChatCompletionStream(t *testing.T) {
	ts := newTestServer("")
	defer ts.Close()

	body := `{"model":"qwen3.5-4b","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hello there friend"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type: got %q", ct)
	}

	var roleSeen, doneSeen, usageSeen, finishSeen bool
	var content strings.Builder
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			doneSeen = true
			continue
		}
		var chunk api.ChatCompletionChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk JSON %q: %v", payload, err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("object: got %q", chunk.Object)
		}
		c := chunk.Choices[0]
		if c.Delta.Role == "assistant" {
			roleSeen = true
		}
		content.WriteString(c.Delta.Content)
		if c.FinishReason != nil {
			finishSeen = true
		}
		if chunk.Usage != nil {
			usageSeen = true
		}
	}
	if !roleSeen {
		t.Error("missing role chunk")
	}
	if !finishSeen {
		t.Error("missing finish_reason chunk")
	}
	if !usageSeen {
		t.Error("missing usage chunk (include_usage=true)")
	}
	if !doneSeen {
		t.Error("missing [DONE] sentinel")
	}
	if content.Len() == 0 {
		t.Error("no streamed content")
	}
}

func TestCompletionNonStream(t *testing.T) {
	ts := newTestServer("")
	defer ts.Close()

	body := `{"model":"qwen3.5-4b","prompt":"once upon a time"}`
	resp, err := http.Post(ts.URL+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out api.CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "text_completion" || out.Choices[0].Text == "" {
		t.Fatalf("unexpected: %+v", out)
	}
}

func TestModelsAndHealth(t *testing.T) {
	ts := newTestServer("")
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var models api.ModelsResponse
	_ = json.NewDecoder(resp.Body).Decode(&models)
	if len(models.Data) != 1 || models.Data[0].ID != "qwen3.5-4b" {
		t.Fatalf("models: %+v", models)
	}

	h, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer h.Body.Close()
	if h.StatusCode != http.StatusOK {
		t.Errorf("health status: %d", h.StatusCode)
	}
}

func TestAuth(t *testing.T) {
	ts := newTestServer("secret-key")
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hi"}]}`
	// No key -> 401.
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("missing key should be 401, got %d", resp.StatusCode)
	}
	// Correct key -> 200.
	req, _ := http.NewRequest("POST", ts.URL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer secret-key")
	req.Header.Set("Content-Type", "application/json")
	resp2, _ := http.DefaultClient.Do(req)
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Errorf("valid key should be 200, got %d", resp2.StatusCode)
	}
	// Health is public even with auth on.
	h, _ := http.Get(ts.URL + "/health")
	h.Body.Close()
	if h.StatusCode != http.StatusOK {
		t.Errorf("health should be public, got %d", h.StatusCode)
	}
}

func TestEmptyMessages(t *testing.T) {
	ts := newTestServer("")
	defer ts.Close()
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("empty messages should be 400, got %d", resp.StatusCode)
	}
}

func TestStreamEncoderEscaping(t *testing.T) {
	var buf bytes.Buffer
	bw := bufio.NewWriter(&buf)
	enc := api.NewStreamEncoder(bw, "id-1", "m", 0)
	if err := enc.Content("line1\nline2 \"quoted\"", false); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, `\n`) || !strings.Contains(out, `\"quoted\"`) {
		t.Errorf("escaping failed: %s", out)
	}
	// The chunk must be valid JSON after the data: prefix.
	payload := strings.TrimSpace(strings.TrimPrefix(out, "data: "))
	var chunk api.ChatCompletionChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		t.Fatalf("encoder produced invalid JSON: %v", err)
	}
	if chunk.Choices[0].Delta.Content != "line1\nline2 \"quoted\"" {
		t.Errorf("round-trip mismatch: %q", chunk.Choices[0].Delta.Content)
	}
}
