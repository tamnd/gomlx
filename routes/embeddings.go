// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"

	"github.com/tamnd/gomlx/api"
	"github.com/tamnd/gomlx/engine"
)

// Embeddings handles POST /v1/embeddings. It mirrors the OpenAI contract: four
// input shapes, optional per-vector dimension truncation with L2 renormalization
// (MRL semantics), and float or base64 encoding. When the server runs without an
// embedding model the backend is nil and the route reports 503, the same way it
// would if the embedding backend were unavailable, rather than returning made-up
// vectors.
func (d *Deps) Embeddings(w http.ResponseWriter, r *http.Request) {
	if d.Embedder == nil {
		writeError(w, http.StatusServiceUnavailable, "embeddings_not_configured",
			"embeddings backend not configured; start the server with an embedding model")
		return
	}

	var req api.EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "malformed request body")
		return
	}

	// The server loads a single embedding model, so requests are locked to it.
	locked := d.Embedder.ModelName()
	if req.Model != "" && locked != "" && req.Model != locked {
		writeError(w, http.StatusBadRequest, "invalid_request", fmt.Sprintf(
			"embedding model %q is not available; this server serves %q for embeddings",
			req.Model, locked))
		return
	}

	texts, tokenBatches, err := parseEmbeddingInput(req.Input)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if req.Dimensions != nil && *req.Dimensions < 1 {
		writeError(w, http.StatusBadRequest, "invalid_request", "dimensions must be a positive integer")
		return
	}

	var (
		vectors      [][]float32
		promptTokens int
	)
	if tokenBatches != nil {
		// For pre-tokenized input trust the caller's count, capped the same way
		// the backend truncates before pooling.
		for _, b := range tokenBatches {
			promptTokens += min(len(b), engine.MaxEmbeddingTokens)
		}
		vectors, err = d.Embedder.EmbedTokens(r.Context(), tokenBatches)
	} else {
		promptTokens = d.Embedder.CountTokens(texts)
		vectors, err = d.Embedder.Embed(r.Context(), texts)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "embedding_error", err.Error())
		return
	}

	if req.Dimensions != nil {
		vectors, err = truncateEmbeddings(vectors, *req.Dimensions)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
	}

	data := make([]api.EmbeddingData, len(vectors))
	useBase64 := req.EncodingFormat == "base64"
	for i, vec := range vectors {
		var payload any
		if useBase64 {
			payload = encodeBase64Vector(vec)
		} else {
			payload = vec
		}
		data[i] = api.EmbeddingData{Object: "embedding", Index: i, Embedding: payload}
	}

	writeJSON(w, http.StatusOK, api.EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  locked,
		Usage:  api.EmbeddingUsage{PromptTokens: promptTokens, TotalTokens: promptTokens},
	})
}

// parseEmbeddingInput resolves the four OpenAI input shapes from raw JSON. It
// returns either texts or token batches, never both. The int forms are kept off
// the text path because the token id 123 embeds differently from the word "123".
func parseEmbeddingInput(raw json.RawMessage) (texts []string, tokenBatches [][]int, err error) {
	if len(raw) == 0 {
		return nil, nil, errors.New("input is required")
	}

	// A single string.
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []string{s}, nil, nil
	}

	// Anything else must be a non-empty array; peek at the first element to tell
	// the three array shapes apart, then decode the whole array into that type so
	// a mixed array fails rather than embedding a subset.
	var elems []json.RawMessage
	if json.Unmarshal(raw, &elems) != nil {
		return nil, nil, errors.New("input must be a string, list of strings, list of integers, or list of lists of integers")
	}
	if len(elems) == 0 {
		return nil, nil, errors.New("input must not be empty")
	}

	switch firstNonSpace(elems[0]) {
	case '"':
		var arr []string
		if json.Unmarshal(raw, &arr) != nil {
			return nil, nil, errors.New("input list must contain only strings")
		}
		return arr, nil, nil
	case '[':
		var arr [][]int
		if json.Unmarshal(raw, &arr) != nil {
			return nil, nil, errors.New("input token lists must contain only integers")
		}
		if err := rejectEmptyBatches(arr); err != nil {
			return nil, nil, err
		}
		return nil, arr, nil
	default:
		var arr []int
		if json.Unmarshal(raw, &arr) != nil {
			return nil, nil, errors.New("input token list must contain only integers")
		}
		if len(arr) == 0 {
			return nil, nil, errors.New("input must not contain empty token sequences")
		}
		return nil, [][]int{arr}, nil
	}
}

// rejectEmptyBatches refuses any empty token sequence. An empty row produces a
// zero-width or all-masked input that pools to a meaningless vector, so it is
// better to fail loudly than to ship garbage to a vector store.
func rejectEmptyBatches(batches [][]int) error {
	for _, b := range batches {
		if len(b) == 0 {
			return errors.New("input must not contain empty token sequences")
		}
	}
	return nil
}

// firstNonSpace returns the first non-whitespace byte of raw, or 0 when raw is
// all whitespace or empty.
func firstNonSpace(raw json.RawMessage) byte {
	for _, c := range raw {
		switch c {
		case ' ', '\t', '\n', '\r':
			continue
		default:
			return c
		}
	}
	return 0
}

// truncateEmbeddings slices each vector to dim and L2-renormalizes it, so the
// truncated vector stays unit-norm and valid for cosine similarity (OpenAI MRL
// semantics). Requesting more than the model's native size is an error.
func truncateEmbeddings(vectors [][]float32, dim int) ([][]float32, error) {
	if len(vectors) == 0 {
		return vectors, nil
	}
	full := len(vectors[0])
	if dim > full {
		return nil, fmt.Errorf("dimensions=%d exceeds the model's embedding size of %d", dim, full)
	}
	out := make([][]float32, len(vectors))
	for i, vec := range vectors {
		sliced := vec[:dim]
		var sum float64
		for _, x := range sliced {
			sum += float64(x) * float64(x)
		}
		norm := math.Sqrt(sum)
		renorm := make([]float32, dim)
		for j, x := range sliced {
			if norm > 0 {
				renorm[j] = float32(float64(x) / norm)
			} else {
				renorm[j] = x
			}
		}
		out[i] = renorm
	}
	return out, nil
}

// encodeBase64Vector packs a vector as little-endian float32 bytes and base64-
// encodes them, the OpenAI base64 encoding format. It saves bandwidth on large
// batches and is the default for several OpenAI client SDKs.
func encodeBase64Vector(vec []float32) string {
	buf := make([]byte, 4*len(vec))
	for i, x := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(x))
	}
	return base64.StdEncoding.EncodeToString(buf)
}
