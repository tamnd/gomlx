// SPDX-License-Identifier: Apache-2.0

package compute

// Llama and Mistral are the same pre-norm transformer as Qwen3 without the
// per-head query/key norm, so they run on the shared DenseModel. The two differ
// from each other only in their default rope base, and only for configs that
// omit rope_theta; every shipped checkpoint sets it explicitly.
var (
	llamaDefaults = archDefaults{
		arch:      "llama",
		qkNorm:    false,
		ropeTheta: 10000.0,
		rmsEps:    1e-5,
	}
	mistralDefaults = archDefaults{
		arch:      "mistral",
		qkNorm:    false,
		ropeTheta: 1000000.0,
		rmsEps:    1e-5,
	}
)

// LoadLlamaArgs parses a Llama or Mistral config.json and fills in the family
// defaults. The two share a config layout and a forward pass; the model_type
// only selects which rope base applies when the config leaves it out.
func LoadLlamaArgs(configJSON []byte) (DenseArgs, error) {
	d := llamaDefaults
	if peekModelType(configJSON) == "mistral" {
		d = mistralDefaults
	}
	return loadDenseArgs(configJSON, d)
}

// NewLlamaModel assembles a runnable Llama or Mistral model from a loaded
// safetensors file and parsed args. It is a thin wrapper over the shared dense
// loader, kept as a named entry point for callers and tests.
func NewLlamaModel(st *SafeTensors, args DenseArgs) (*DenseModel, error) {
	return newDenseModel(st, args)
}
