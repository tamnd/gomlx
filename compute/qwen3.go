// SPDX-License-Identifier: Apache-2.0

package compute

// qwen3Defaults are the Qwen3 family fallbacks: a per-head query/key norm, a 1e6
// rope base, and a 1e-6 norm epsilon. Qwen3 was the first forward pass the
// backend targeted because its small checkpoints fit in the available memory.
var qwen3Defaults = archDefaults{
	arch:      "qwen3",
	qkNorm:    true,
	ropeTheta: 1000000.0,
	rmsEps:    1e-6,
}

// LoadQwen3Args parses a Qwen3 config.json and fills in the Qwen3 defaults.
func LoadQwen3Args(configJSON []byte) (DenseArgs, error) {
	return loadDenseArgs(configJSON, qwen3Defaults)
}

// NewQwen3Model assembles a runnable Qwen3 model from a loaded safetensors file
// and parsed args. It is a thin wrapper over the shared dense loader, kept as a
// named entry point for callers and tests that target Qwen3 specifically.
func NewQwen3Model(st *SafeTensors, args DenseArgs) (*DenseModel, error) {
	return newDenseModel(st, args)
}
