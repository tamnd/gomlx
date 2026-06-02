// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/json"
	"strings"
)

// The Anthropic Messages API differs from OpenAI chat completions in both the
// request and the response shape: content is a list of typed blocks, tools use
// an input_schema, and the streaming wire format is a sequence of named SSE
// events rather than chat-completion chunks. These types model that surface;
// the route translates to and from the OpenAI-shaped engine path.

// AnthropicContentBlock is one block in a request message. The fields cover the
// text, tool_use, tool_result, and image variants; only those relevant to a
// given Type are populated.
type AnthropicContentBlock struct {
	Type string `json:"type"`
	// text
	Text string `json:"text,omitempty"`
	// tool_use
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`
	// tool_result
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   *bool           `json:"is_error,omitempty"`
	// image
	Source map[string]any `json:"source,omitempty"`
}

// AnthropicMessage is one conversation turn. Content is either a plain string
// or an array of content blocks, so it is decoded lazily.
type AnthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// Blocks normalizes the message content to a list of blocks. A bare string
// becomes a single text block.
func (m AnthropicMessage) Blocks() []AnthropicContentBlock {
	if len(m.Content) == 0 {
		return nil
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return []AnthropicContentBlock{{Type: "text", Text: s}}
	}
	var blocks []AnthropicContentBlock
	_ = json.Unmarshal(m.Content, &blocks)
	return blocks
}

// AnthropicToolDef is a tool offered in Anthropic format.
type AnthropicToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
}

// AnthropicRequest is the POST /v1/messages body. max_tokens is required by the
// Anthropic API. system is a string or a list of text blocks.
type AnthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []AnthropicMessage `json:"messages"`
	System        json.RawMessage    `json:"system,omitempty"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Tools         []AnthropicToolDef `json:"tools,omitempty"`
	ToolChoice    map[string]any     `json:"tool_choice,omitempty"`
	Metadata      map[string]any     `json:"metadata,omitempty"`
}

// SystemText flattens the system field to plain text, joining list blocks with
// newlines.
func (r *AnthropicRequest) SystemText() string {
	if len(r.System) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(r.System, &s); err == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(r.System, &blocks); err != nil {
		return ""
	}
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// AnthropicUsage is token accounting in the Anthropic shape.
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// AnthropicResponseBlock is one block in a non-streaming response. Text and
// Thinking are pointers so an empty text block still serializes its key while a
// tool_use block omits both.
type AnthropicResponseBlock struct {
	Type     string          `json:"type"`
	Text     *string         `json:"text,omitempty"`
	Thinking *string         `json:"thinking,omitempty"`
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name,omitempty"`
	Input    json.RawMessage `json:"input,omitempty"`
}

// AnthropicResponse is the non-streaming POST /v1/messages response.
type AnthropicResponse struct {
	ID           string                   `json:"id"`
	Type         string                   `json:"type"`
	Role         string                   `json:"role"`
	Model        string                   `json:"model"`
	Content      []AnthropicResponseBlock `json:"content"`
	StopReason   string                   `json:"stop_reason"`
	StopSequence *string                  `json:"stop_sequence"`
	Usage        AnthropicUsage           `json:"usage"`
}
