// SPDX-License-Identifier: Apache-2.0

// Package routes implements the HTTP handlers for the OpenAI- and Anthropic-
// compatible endpoints. Handlers hang off Deps so the server package can wire
// them with a concrete engine.
package routes

import (
	"encoding/json"
	"net/http"

	"github.com/tamnd/gomlx/api"
	"github.com/tamnd/gomlx/engine"
	"github.com/tamnd/gomlx/mcp"
)

// Deps carries the shared dependencies the route handlers need.
type Deps struct {
	Engine           engine.Engine
	Model            string
	DefaultMaxTokens int

	// ToolCallParser names the wire format used to extract tool calls from
	// model output. Empty falls back to the auto parser.
	ToolCallParser string

	// MCP pools the tools from the configured MCP servers. It is nil when the
	// server was started without an MCP config, in which case the MCP routes
	// report an empty, unconfigured subsystem.
	MCP *mcp.Manager

	// Embedder serves the embeddings endpoint. It is nil when the server was
	// started without an embedding model, in which case /v1/embeddings reports
	// the subsystem as unconfigured.
	Embedder engine.Embedder

	// Cancels tracks in-flight streaming requests so they can be cancelled by
	// id. Its zero value is ready to use.
	Cancels cancelRegistry
}

// writeJSON serializes v as JSON with the given status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError emits an OpenAI-style error envelope.
func writeError(w http.ResponseWriter, status int, typ, msg string) {
	writeJSON(w, status, api.ErrorBody{Error: api.ErrorDetail{Message: msg, Type: typ}})
}

// toChatMessages converts API messages to engine messages. A message's content
// may be a plain string or an array of multimodal parts; TextContent flattens
// either form to the text the engine consumes. Image parts ride along on the
// API message and are picked up by the multimodal stage rather than dropped
// here on the text path.
func toChatMessages(msgs []api.Message) []engine.ChatMessage {
	out := make([]engine.ChatMessage, len(msgs))
	for i, m := range msgs {
		out[i] = engine.ChatMessage{
			Role:       m.Role,
			Content:    m.TextContent(),
			ToolCallID: m.ToolCallID,
			Name:       m.Name,
		}
	}
	return out
}

// resolveSampling builds SamplingParams from optional request pointers, applying
// the server default for max tokens when the client omits it.
func (d *Deps) resolveSampling(maxTokens *int, temp, topP, minP, repPen, presPen, freqPen *float64, topK *int, stop []string) engine.SamplingParams {
	p := engine.DefaultSamplingParams()
	p.MaxTokens = d.DefaultMaxTokens
	if p.MaxTokens == 0 {
		p.MaxTokens = 256
	}
	if maxTokens != nil {
		p.MaxTokens = *maxTokens
	}
	if temp != nil {
		p.Temperature = *temp
	}
	if topP != nil {
		p.TopP = *topP
	}
	if minP != nil {
		p.MinP = *minP
	}
	if topK != nil {
		p.TopK = *topK
	}
	if repPen != nil {
		p.RepetitionPenalty = *repPen
	}
	if presPen != nil {
		p.PresencePenalty = *presPen
	}
	if freqPen != nil {
		p.FrequencyPenalty = *freqPen
	}
	if stop != nil {
		p.Stop = stop
	}
	return p
}
