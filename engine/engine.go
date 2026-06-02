// SPDX-License-Identifier: Apache-2.0

package engine

import "context"

// Channel labels the semantic stream a GenerationOutput belongs to.
const (
	ChannelContent   = "content"
	ChannelReasoning = "reasoning"
	ChannelToolCall  = "tool_call"
)

// GenerationOutput is the result of a generation step, compatible with both
// the non-streaming (complete text) and streaming (incremental NewText) paths.
// Ported from engine/base.py GenerationOutput.
type GenerationOutput struct {
	Text             string
	Tokens           []int
	PromptTokens     int
	CompletionTokens int
	FinishReason     string

	// Streaming fields.
	NewText  string
	Finished bool

	// Channel is "content", "reasoning", "tool_call", or "".
	Channel string

	// RawText preserves pre-cleaning output so a reasoning parser can still see
	// channel markers that text cleaning strips. ReasoningText is the
	// token-level reasoning extraction when the engine populates it.
	RawText       string
	ReasoningText string
}

// ChatMessage is one message in a chat request. Content is the plain-text form;
// multimodal content parts are added in the multimodal stage.
type ChatMessage struct {
	Role       string
	Content    string
	ToolCalls  []any
	ToolCallID string
	Name       string
}

// Engine is the inference backend interface. The mock backend (stage 2) and the
// MLX-backed engine (stage 4) both implement it. Streaming methods return a
// receive-only channel that is closed when generation finishes; the final
// element has Finished=true. Ported from engine/base.py BaseEngine.
type Engine interface {
	ModelName() string
	IsMLLM() bool

	Start(ctx context.Context) error
	Stop(ctx context.Context) error

	Generate(ctx context.Context, prompt string, p SamplingParams) (GenerationOutput, error)
	StreamGenerate(ctx context.Context, prompt string, p SamplingParams) (<-chan GenerationOutput, error)

	Chat(ctx context.Context, msgs []ChatMessage, p SamplingParams, tools []any) (GenerationOutput, error)
	StreamChat(ctx context.Context, msgs []ChatMessage, p SamplingParams, tools []any) (<-chan GenerationOutput, error)

	// EstimateNewTokens returns (promptTokens, freshTokens) for admission and
	// cloud-routing decisions.
	EstimateNewTokens(prompt string) (promptTokens, freshTokens int)
}
