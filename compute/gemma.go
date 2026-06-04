// SPDX-License-Identifier: Apache-2.0

package compute

// Gemma (the first Gemma release, model_type "gemma") is the Llama-shaped dense
// decoder with three changes. The token embeddings are scaled by
// sqrt(hidden_size) before the first layer; the MLP gate uses the exact GELU
// rather than SiLU; and every RMSNorm applies its weight as (1 + weight).
// Attention is otherwise the standard grouped-query form, with an independent
// head_dim, no projection bias, and no per-head query/key norm. The rope base
// and norm epsilon match the values Gemma ships, and the word embeddings are
// tied, so the embedding matrix doubles as the output projection.
var gemmaDefaults = archDefaults{
	arch:        "gemma",
	qkNorm:      false,
	attnBias:    false,
	ropeTheta:   10000.0,
	rmsEps:      1e-6,
	embedScale:  true,
	geGLU:       true,
	normOnePlus: true,
}

// LoadGemmaArgs parses a Gemma config.json and fills in the Gemma defaults.
func LoadGemmaArgs(configJSON []byte) (DenseArgs, error) {
	return loadDenseArgs(configJSON, gemmaDefaults)
}

// NewGemmaModel assembles a runnable Gemma model from a loaded safetensors file
// and parsed args. It is a thin wrapper over the shared dense loader, kept as a
// named entry point for callers and tests that target Gemma specifically; the
// embedding scale, GELU gate, and (1 + weight) norms follow from the flags the
// loader sets.
func NewGemmaModel(st *SafeTensors, args DenseArgs) (*DenseModel, error) {
	return newDenseModel(st, args)
}
