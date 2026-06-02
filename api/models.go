// SPDX-License-Identifier: Apache-2.0

// Package api holds OpenAI- and Anthropic-compatible request/response types and
// the streaming encoder.
package api

// ImageURL is an OpenAI image content reference.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ContentPart is one element of a multimodal message content array.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// FunctionCall is the function portion of a tool call.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// ToolCall is an assistant tool invocation.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// ToolDefinition is a tool the client offers to the model.
type ToolDefinition struct {
	Type     string         `json:"type"`
	Function map[string]any `json:"function"`
}

// Message is a chat message. Content may be a string or an array of parts; it is
// kept as json.RawMessage-free `any` and normalized in the route.
type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// StreamOptions controls streaming extras.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatCompletionRequest is the OpenAI /v1/chat/completions request body.
type ChatCompletionRequest struct {
	Model               string           `json:"model"`
	Messages            []Message        `json:"messages"`
	Temperature         *float64         `json:"temperature,omitempty"`
	TopP                *float64         `json:"top_p,omitempty"`
	MaxTokens           *int             `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int             `json:"max_completion_tokens,omitempty"`
	Stream              bool             `json:"stream,omitempty"`
	StreamOptions       *StreamOptions   `json:"stream_options,omitempty"`
	Stop                []string         `json:"stop,omitempty"`
	TopK                *int             `json:"top_k,omitempty"`
	MinP                *float64         `json:"min_p,omitempty"`
	RepetitionPenalty   *float64         `json:"repetition_penalty,omitempty"`
	PresencePenalty     *float64         `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64         `json:"frequency_penalty,omitempty"`
	Tools               []ToolDefinition `json:"tools,omitempty"`
	ToolChoice          any              `json:"tool_choice,omitempty"`
	EnableThinking      *bool            `json:"enable_thinking,omitempty"`
	N                   *int             `json:"n,omitempty"`
}

// AssistantMessage is the assistant reply in a non-streaming response.
type AssistantMessage struct {
	Role             string     `json:"role"`
	Content          *string    `json:"content"`
	ReasoningContent *string    `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// ChatCompletionChoice is one choice in a non-streaming chat response.
type ChatCompletionChoice struct {
	Index        int              `json:"index"`
	Message      AssistantMessage `json:"message"`
	FinishReason *string          `json:"finish_reason"`
}

// Usage is token accounting.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse is the non-streaming chat response.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   Usage                  `json:"usage"`
}

// ChatCompletionChunkDelta is the incremental delta in a streaming chunk.
type ChatCompletionChunkDelta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// ChatCompletionChunkChoice is one choice in a streaming chunk.
type ChatCompletionChunkChoice struct {
	Index        int                      `json:"index"`
	Delta        ChatCompletionChunkDelta `json:"delta"`
	FinishReason *string                  `json:"finish_reason"`
}

// ChatCompletionChunk is a single SSE chunk.
type ChatCompletionChunk struct {
	ID      string                      `json:"id"`
	Object  string                      `json:"object"`
	Created int64                       `json:"created"`
	Model   string                      `json:"model"`
	Choices []ChatCompletionChunkChoice `json:"choices"`
	Usage   *Usage                      `json:"usage,omitempty"`
}

// CompletionRequest is the OpenAI /v1/completions request body.
type CompletionRequest struct {
	Model             string   `json:"model"`
	Prompt            any      `json:"prompt"` // string or []string
	Temperature       *float64 `json:"temperature,omitempty"`
	TopP              *float64 `json:"top_p,omitempty"`
	MaxTokens         *int     `json:"max_tokens,omitempty"`
	Stream            bool     `json:"stream,omitempty"`
	Stop              []string `json:"stop,omitempty"`
	TopK              *int     `json:"top_k,omitempty"`
	MinP              *float64 `json:"min_p,omitempty"`
	RepetitionPenalty *float64 `json:"repetition_penalty,omitempty"`
	PresencePenalty   *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty  *float64 `json:"frequency_penalty,omitempty"`
}

// CompletionChoice is one choice in a text completion response.
type CompletionChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

// CompletionResponse is the non-streaming text completion response.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`
}

// ModelInfo describes one served model.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse is the /v1/models list.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ErrorBody is the OpenAI-style error envelope.
type ErrorBody struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the error payload.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}
