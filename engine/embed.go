// SPDX-License-Identifier: Apache-2.0

package engine

import "context"

// MaxEmbeddingTokens caps how many tokens of a single input contribute to the
// reported prompt-token count, matching the truncation the embedding backend
// applies before pooling.
const MaxEmbeddingTokens = 512

// Embedder produces sentence embeddings. It is a separate backend from Engine:
// a server can serve generation, embeddings, both, or neither. The MLX-backed
// implementation lands with the compute backend; until then the embeddings
// route reports the subsystem as unconfigured rather than returning fabricated
// vectors. Ported from the embedding-engine surface the reference route calls.
type Embedder interface {
	// ModelName is the embedding model this backend serves. The route locks
	// requests to it, since the server loads a single embedding model.
	ModelName() string

	// Embed returns one vector per input text.
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// EmbedTokens returns one vector per pre-tokenized input, letting callers
	// that share a tokenizer skip re-tokenization.
	EmbedTokens(ctx context.Context, batches [][]int) ([][]float32, error)

	// CountTokens reports the total prompt tokens across texts, used for usage
	// accounting.
	CountTokens(texts []string) int
}
