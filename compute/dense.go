// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"encoding/json"
	"fmt"
)

// DenseArgs holds the architecture hyperparameters shared by the dense decoder
// families this backend runs: Qwen3, Qwen2, Llama, and Mistral. They are the
// same pre-norm transformer with grouped-query attention, rotary embeddings, and
// a SwiGLU MLP; they differ only in a couple of capabilities and a few default
// values. Arch names the family, QKNorm records whether the attention applies a
// per-head query/key RMSNorm (Qwen3 does, the rest do not), and AttentionBias
// records whether the query/key/value projections carry a bias (Qwen2 does, the
// rest do not). Values absent from a given config.json fall back to the family
// defaults supplied by the arch-specific loader.
type DenseArgs struct {
	Arch   string // "qwen3", "qwen2", "llama", or "mistral"
	QKNorm bool   // per-head query/key RMSNorm before RoPE

	ModelType             string
	HiddenSize            int
	IntermediateSize      int
	NumHiddenLayers       int
	NumAttentionHeads     int
	NumKeyValueHeads      int
	HeadDim               int
	RMSNormEps            float64
	RopeTheta             float64
	VocabSize             int
	MaxPositionEmbeddings int
	TieWordEmbeddings     bool
	AttentionBias         bool
	SlidingWindow         int
	EOSTokenIDs           []int // parsed from eos_token_id (a scalar or a list)
}

// archDefaults carries the family-specific fallbacks a config.json may omit. Real
// checkpoints set rope_theta and rms_norm_eps explicitly, so these matter mainly
// for terse or older configs.
type archDefaults struct {
	arch      string
	qkNorm    bool
	attnBias  bool // query/key/value projections carry a bias (Qwen2)
	ropeTheta float64
	rmsEps    float64
}

// rawDense distinguishes "key absent" from "key present and zero" for the fields
// that have non-zero defaults, by decoding them through pointers. eos_token_id is
// kept raw because Hugging Face writes it as either a single id or a list.
type rawDense struct {
	ModelType             string          `json:"model_type"`
	HiddenSize            int             `json:"hidden_size"`
	IntermediateSize      int             `json:"intermediate_size"`
	NumHiddenLayers       int             `json:"num_hidden_layers"`
	NumAttentionHeads     int             `json:"num_attention_heads"`
	NumKeyValueHeads      *int            `json:"num_key_value_heads"`
	HeadDim               *int            `json:"head_dim"`
	RMSNormEps            *float64        `json:"rms_norm_eps"`
	RopeTheta             *float64        `json:"rope_theta"`
	VocabSize             int             `json:"vocab_size"`
	MaxPositionEmbeddings int             `json:"max_position_embeddings"`
	TieWordEmbeddings     bool            `json:"tie_word_embeddings"`
	AttentionBias         bool            `json:"attention_bias"`
	SlidingWindow         int             `json:"sliding_window"`
	EOSTokenID            json.RawMessage `json:"eos_token_id"`
}

// loadDenseArgs parses a config.json into DenseArgs, applying the family defaults
// for any field the config leaves out. It validates the dimensions so a bad
// config fails at load rather than midway through a forward pass.
func loadDenseArgs(configJSON []byte, d archDefaults) (DenseArgs, error) {
	var r rawDense
	if err := json.Unmarshal(configJSON, &r); err != nil {
		return DenseArgs{}, fmt.Errorf("%s: parse config: %w", d.arch, err)
	}

	a := DenseArgs{
		Arch:                  d.arch,
		QKNorm:                d.qkNorm,
		ModelType:             r.ModelType,
		HiddenSize:            r.HiddenSize,
		IntermediateSize:      r.IntermediateSize,
		NumHiddenLayers:       r.NumHiddenLayers,
		NumAttentionHeads:     r.NumAttentionHeads,
		VocabSize:             r.VocabSize,
		MaxPositionEmbeddings: r.MaxPositionEmbeddings,
		TieWordEmbeddings:     r.TieWordEmbeddings,
		AttentionBias:         r.AttentionBias,
		SlidingWindow:         r.SlidingWindow,
		EOSTokenIDs:           parseEOS(r.EOSTokenID),
	}

	// num_key_value_heads defaults to num_attention_heads (no grouped-query).
	if r.NumKeyValueHeads != nil {
		a.NumKeyValueHeads = *r.NumKeyValueHeads
	} else {
		a.NumKeyValueHeads = r.NumAttentionHeads
	}

	// head_dim defaults to hidden_size / num_attention_heads.
	if r.HeadDim != nil {
		a.HeadDim = *r.HeadDim
	} else if r.NumAttentionHeads > 0 {
		a.HeadDim = r.HiddenSize / r.NumAttentionHeads
	}

	if r.RMSNormEps != nil {
		a.RMSNormEps = *r.RMSNormEps
	} else {
		a.RMSNormEps = d.rmsEps
	}

	if r.RopeTheta != nil {
		a.RopeTheta = *r.RopeTheta
	} else {
		a.RopeTheta = d.ropeTheta
	}

	// Qwen2 always biases the attention projections; the field is part of the
	// architecture rather than the config, so the family default forces it on
	// even when the config omits attention_bias.
	if d.attnBias {
		a.AttentionBias = true
	}

	if err := a.validate(); err != nil {
		return DenseArgs{}, err
	}
	return a, nil
}

func (a DenseArgs) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("%s: hidden_size must be positive, got %d", a.Arch, a.HiddenSize)
	case a.NumHiddenLayers <= 0:
		return fmt.Errorf("%s: num_hidden_layers must be positive, got %d", a.Arch, a.NumHiddenLayers)
	case a.NumAttentionHeads <= 0:
		return fmt.Errorf("%s: num_attention_heads must be positive, got %d", a.Arch, a.NumAttentionHeads)
	case a.VocabSize <= 0:
		return fmt.Errorf("%s: vocab_size must be positive, got %d", a.Arch, a.VocabSize)
	case a.NumAttentionHeads%a.NumKeyValueHeads != 0:
		return fmt.Errorf("%s: num_attention_heads %d not divisible by num_key_value_heads %d",
			a.Arch, a.NumAttentionHeads, a.NumKeyValueHeads)
	}
	return nil
}

// LoadArgs parses a config.json for any supported dense architecture, choosing
// the family by its model_type. An unsupported or missing type returns an error
// naming it, so the caller can fall back to the mock engine rather than load a
// model it cannot run correctly.
func LoadArgs(configJSON []byte) (DenseArgs, error) {
	mt := peekModelType(configJSON)
	switch mt {
	case "qwen3":
		return LoadQwen3Args(configJSON)
	case "llama", "mistral":
		return LoadLlamaArgs(configJSON)
	case "":
		return DenseArgs{}, fmt.Errorf("compute: config.json has no model_type")
	default:
		return DenseArgs{}, fmt.Errorf("compute: unsupported model_type %q", mt)
	}
}

// peekModelType reads just the model_type field, ignoring everything else, so the
// dispatcher can route a config without committing to a full parse.
func peekModelType(configJSON []byte) string {
	var probe struct {
		ModelType string `json:"model_type"`
	}
	_ = json.Unmarshal(configJSON, &probe)
	return probe.ModelType
}

// parseEOS reads eos_token_id, which Hugging Face writes as either a single id or
// a list of ids. An absent or unparseable field yields nil, leaving the caller to
// fall back to a default end token.
func parseEOS(raw json.RawMessage) []int {
	if len(raw) == 0 {
		return nil
	}
	var one int
	if err := json.Unmarshal(raw, &one); err == nil {
		return []int{one}
	}
	var many []int
	if err := json.Unmarshal(raw, &many); err == nil {
		return many
	}
	return nil
}
