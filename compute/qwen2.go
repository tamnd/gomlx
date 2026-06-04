// SPDX-License-Identifier: Apache-2.0

package compute

// Qwen2 (and Qwen2.5, which ships under the same model_type) is the Llama
// architecture without the per-head query/key norm but with a bias on the
// query, key, and value projections. The output projection stays unbiased. The
// bias is part of the architecture rather than the config, so attnBias is forced
// on here even when a config omits attention_bias. The rope base and norm
// epsilon match the values Qwen ships.
var qwen2Defaults = archDefaults{
	arch:      "qwen2",
	qkNorm:    false,
	attnBias:  true,
	ropeTheta: 1000000.0,
	rmsEps:    1e-6,
}

// LoadQwen2Args parses a Qwen2 or Qwen2.5 config.json and fills in the Qwen2
// defaults.
func LoadQwen2Args(configJSON []byte) (DenseArgs, error) {
	return loadDenseArgs(configJSON, qwen2Defaults)
}

// NewQwen2Model assembles a runnable Qwen2 or Qwen2.5 model from a loaded
// safetensors file and parsed args. It is a thin wrapper over the shared dense
// loader, kept as a named entry point for callers and tests that target Qwen2
// specifically.
func NewQwen2Model(st *SafeTensors, args DenseArgs) (*DenseModel, error) {
	return newDenseModel(st, args)
}
