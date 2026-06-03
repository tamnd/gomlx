// SPDX-License-Identifier: Apache-2.0

// Package cloudrouter sends the requests that are expensive to serve locally to
// an OpenAI-compatible cloud endpoint instead. Prefill on Apple Silicon grows
// with the square of the prompt length, so a cold request with a very long
// prompt can dominate a machine that is otherwise serving many short ones. The
// router keeps the short, cache-friendly traffic local and forwards only the
// requests whose count of new, uncached tokens crosses a threshold, where the
// cloud's batching and bigger machines win.
package cloudrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Config configures a Router. BaseURL is an OpenAI-compatible root such as
// https://api.openai.com/v1; the router appends /chat/completions. Threshold is
// the count of new tokens above which a request is sent to the cloud; a
// non-positive threshold disables routing so everything stays local.
type Config struct {
	Model     string
	Threshold int
	BaseURL   string
	APIKey    string
	Client    *http.Client
}

// Message is one OpenAI-format chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Params carries the optional sampling and tool fields to forward. Pointer fields
// are omitted from the upstream request when nil, so the cloud provider applies
// its own defaults rather than receiving a zero value the caller never set.
type Params struct {
	Temperature      *float64
	TopP             *float64
	FrequencyPenalty *float64
	PresencePenalty  *float64
	MaxTokens        *int
	Stop             []string
	Tools            []any
	ToolChoice       any
	ResponseFormat   any
	Stream           bool
}

// Router decides whether a request should go to the cloud and forwards it when so.
type Router struct {
	cfg    Config
	client *http.Client
}

// New returns a Router for cfg. When no HTTP client is supplied the default
// client is used.
func New(cfg Config) *Router {
	client := cfg.Client
	if client == nil {
		client = http.DefaultClient
	}
	return &Router{cfg: cfg, client: client}
}

// ShouldRoute reports whether a request with newTokens uncached tokens should be
// sent to the cloud. A non-positive threshold keeps everything local.
func (r *Router) ShouldRoute(newTokens int) bool {
	return r.cfg.Threshold > 0 && newTokens > r.cfg.Threshold
}

// buildBody assembles the upstream request body, setting the configured model and
// including only the optional params the caller actually set.
func (r *Router) buildBody(messages []Message, p Params) map[string]any {
	body := map[string]any{
		"model":    r.cfg.Model,
		"messages": messages,
		"stream":   p.Stream,
	}
	if p.Temperature != nil {
		body["temperature"] = *p.Temperature
	}
	if p.TopP != nil {
		body["top_p"] = *p.TopP
	}
	if p.FrequencyPenalty != nil {
		body["frequency_penalty"] = *p.FrequencyPenalty
	}
	if p.PresencePenalty != nil {
		body["presence_penalty"] = *p.PresencePenalty
	}
	if p.MaxTokens != nil {
		body["max_tokens"] = *p.MaxTokens
	}
	if len(p.Stop) > 0 {
		body["stop"] = p.Stop
	}
	if len(p.Tools) > 0 {
		body["tools"] = p.Tools
	}
	if p.ToolChoice != nil {
		body["tool_choice"] = p.ToolChoice
	}
	if p.ResponseFormat != nil {
		body["response_format"] = p.ResponseFormat
	}
	return body
}

// endpoint is the chat-completions URL for the configured base.
func (r *Router) endpoint() string {
	return strings.TrimRight(r.cfg.BaseURL, "/") + "/chat/completions"
}

// Complete forwards a non-streaming chat completion to the cloud endpoint and
// returns the raw response body. The caller relays that body to its own client,
// since it is already in OpenAI format. A non-2xx response is returned as an
// error carrying the upstream status and body.
func (r *Router) Complete(ctx context.Context, messages []Message, p Params) ([]byte, error) {
	p.Stream = false
	raw, err := json.Marshal(r.buildBody(messages, p))
	if err != nil {
		return nil, fmt.Errorf("cloudrouter: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoint(), bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("cloudrouter: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if r.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.cfg.APIKey)
	}

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cloudrouter: request to %s: %w", r.cfg.Model, err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("cloudrouter: read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("cloudrouter: upstream %s returned %d: %s", r.cfg.Model, resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return data, nil
}
