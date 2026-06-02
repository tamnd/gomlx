// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"strings"
	"time"
)

// MockEngine is a GPU-free Engine used to exercise and benchmark the full HTTP
// serving path before the compute backend (stage 4) lands. It produces
// deterministic output derived from the prompt so tests are stable, and streams
// word-by-word with an optional inter-token delay to simulate decode latency.
type MockEngine struct {
	name string
	// PerTokenDelay simulates decode latency in streaming mode. Zero is fastest
	// (useful for overhead benchmarks).
	PerTokenDelay time.Duration
}

// NewMockEngine returns a mock engine reporting the given model name.
func NewMockEngine(name string) *MockEngine { return &MockEngine{name: name} }

func (m *MockEngine) ModelName() string           { return m.name }
func (m *MockEngine) IsMLLM() bool                { return false }
func (m *MockEngine) Start(context.Context) error { return nil }
func (m *MockEngine) Stop(context.Context) error  { return nil }

// EstimateNewTokens uses a coarse whitespace token estimate.
func (m *MockEngine) EstimateNewTokens(prompt string) (int, int) {
	n := len(strings.Fields(prompt))
	return n, n
}

// mockReply builds a deterministic response for a prompt.
func mockReply(prompt string, maxTokens int) string {
	words := strings.Fields(prompt)
	preview := prompt
	if len(words) > 8 {
		preview = strings.Join(words[:8], " ") + "..."
	}
	reply := "Mock response to: " + preview
	out := strings.Fields(reply)
	if maxTokens > 0 && len(out) > maxTokens {
		out = out[:maxTokens]
	}
	return strings.Join(out, " ")
}

func (m *MockEngine) Generate(_ context.Context, prompt string, p SamplingParams) (GenerationOutput, error) {
	text := mockReply(prompt, p.MaxTokens)
	pt, _ := m.EstimateNewTokens(prompt)
	ct := len(strings.Fields(text))
	return GenerationOutput{
		Text:             text,
		PromptTokens:     pt,
		CompletionTokens: ct,
		FinishReason:     "stop",
		Finished:         true,
		Channel:          ChannelContent,
	}, nil
}

func (m *MockEngine) StreamGenerate(ctx context.Context, prompt string, p SamplingParams) (<-chan GenerationOutput, error) {
	text := mockReply(prompt, p.MaxTokens)
	pt, _ := m.EstimateNewTokens(prompt)
	return m.stream(ctx, text, pt), nil
}

func (m *MockEngine) Chat(ctx context.Context, msgs []ChatMessage, p SamplingParams, _ []any) (GenerationOutput, error) {
	return m.Generate(ctx, flatten(msgs), p)
}

func (m *MockEngine) StreamChat(ctx context.Context, msgs []ChatMessage, p SamplingParams, _ []any) (<-chan GenerationOutput, error) {
	return m.StreamGenerate(ctx, flatten(msgs), p)
}

// stream emits one GenerationOutput per word, then a final Finished marker. It
// respects context cancellation so client disconnects stop generation.
func (m *MockEngine) stream(ctx context.Context, text string, promptTokens int) <-chan GenerationOutput {
	ch := make(chan GenerationOutput)
	go func() {
		defer close(ch)
		words := strings.Fields(text)
		for i, w := range words {
			piece := w
			if i > 0 {
				piece = " " + w
			}
			select {
			case <-ctx.Done():
				return
			case ch <- GenerationOutput{NewText: piece, Channel: ChannelContent}:
			}
			if m.PerTokenDelay > 0 {
				select {
				case <-ctx.Done():
					return
				case <-time.After(m.PerTokenDelay):
				}
			}
		}
		select {
		case <-ctx.Done():
		case ch <- GenerationOutput{
			Finished:         true,
			FinishReason:     "stop",
			PromptTokens:     promptTokens,
			CompletionTokens: len(words),
			Channel:          ChannelContent,
		}:
		}
	}()
	return ch
}

func flatten(msgs []ChatMessage) string {
	var b strings.Builder
	for _, msg := range msgs {
		b.WriteString(msg.Content)
		b.WriteByte('\n')
	}
	return b.String()
}

// compile-time check.
var _ Engine = (*MockEngine)(nil)
