// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"encoding/json"
	"fmt"
)

// Qwen3Args holds the architecture hyperparameters for a Qwen3 dense model,
// parsed from a Hugging Face config.json. Qwen3 is the first forward pass the
// backend targets because the small checkpoints fit in the available memory.
// The field set mirrors the keys the reference reads; values absent from a
// given config.json fall back to the documented Qwen3 defaults in
// LoadQwen3Args.
type Qwen3Args struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HeadDim               int     `json:"head_dim"`
	RMSNormEps            float64 `json:"rms_norm_eps"`
	RopeTheta             float64 `json:"rope_theta"`
	VocabSize             int     `json:"vocab_size"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	AttentionBias         bool    `json:"attention_bias"`
	SlidingWindow         int     `json:"sliding_window"`
}

// rawQwen3 distinguishes "key absent" from "key present and zero" for the few
// fields that have non-zero defaults, by using pointers during decode.
type rawQwen3 struct {
	ModelType             string   `json:"model_type"`
	HiddenSize            int      `json:"hidden_size"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumKeyValueHeads      *int     `json:"num_key_value_heads"`
	HeadDim               *int     `json:"head_dim"`
	RMSNormEps            *float64 `json:"rms_norm_eps"`
	RopeTheta             *float64 `json:"rope_theta"`
	VocabSize             int      `json:"vocab_size"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	TieWordEmbeddings     bool     `json:"tie_word_embeddings"`
	AttentionBias         bool     `json:"attention_bias"`
	SlidingWindow         int      `json:"sliding_window"`
}

// LoadQwen3Args parses a Qwen3 config.json and fills in defaults. It validates
// that the required dimensions are present and internally consistent so a bad
// config fails at load rather than midway through a forward pass.
func LoadQwen3Args(configJSON []byte) (Qwen3Args, error) {
	var r rawQwen3
	if err := json.Unmarshal(configJSON, &r); err != nil {
		return Qwen3Args{}, fmt.Errorf("qwen3: parse config: %w", err)
	}

	a := Qwen3Args{
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
		a.RMSNormEps = 1e-6
	}

	if r.RopeTheta != nil {
		a.RopeTheta = *r.RopeTheta
	} else {
		a.RopeTheta = 1000000.0
	}

	if err := a.validate(); err != nil {
		return Qwen3Args{}, err
	}
	return a, nil
}

func (a Qwen3Args) validate() error {
	switch {
	case a.HiddenSize <= 0:
		return fmt.Errorf("qwen3: hidden_size must be positive, got %d", a.HiddenSize)
	case a.NumHiddenLayers <= 0:
		return fmt.Errorf("qwen3: num_hidden_layers must be positive, got %d", a.NumHiddenLayers)
	case a.NumAttentionHeads <= 0:
		return fmt.Errorf("qwen3: num_attention_heads must be positive, got %d", a.NumAttentionHeads)
	case a.VocabSize <= 0:
		return fmt.Errorf("qwen3: vocab_size must be positive, got %d", a.VocabSize)
	case a.NumAttentionHeads%a.NumKeyValueHeads != 0:
		return fmt.Errorf("qwen3: num_attention_heads %d not divisible by num_key_value_heads %d",
			a.NumAttentionHeads, a.NumKeyValueHeads)
	}
	return nil
}
