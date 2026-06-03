// SPDX-License-Identifier: Apache-2.0

package api

import "encoding/json"

// EmbeddingRequest is the OpenAI /v1/embeddings request body. Input is kept as
// raw JSON because the OpenAI spec allows four shapes that must be told apart in
// the route: a single string, a batch of strings, a single pre-tokenized input
// (list of ints), or a batch of pre-tokenized inputs (list of lists of ints).
// The int forms must not be routed through the string path, since the token id
// 123 embeds differently from the word "123".
type EmbeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format,omitempty"`
	Dimensions     *int            `json:"dimensions,omitempty"`
	User           string          `json:"user,omitempty"`
}

// EmbeddingData is one embedding result. Embedding is a []float32 when the
// encoding format is "float" and a base64 string when it is "base64".
type EmbeddingData struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"`
}

// EmbeddingUsage is token accounting for an embeddings request. Embeddings have
// no completion side, so prompt and total tokens are equal.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// EmbeddingResponse is the OpenAI /v1/embeddings response body.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingUsage  `json:"usage"`
}
