// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"reflect"
	"testing"
)

func TestLoadGemmaArgs(t *testing.T) {
	cfg := []byte(`{
		"model_type": "gemma",
		"hidden_size": 2048,
		"intermediate_size": 16384,
		"num_hidden_layers": 18,
		"num_attention_heads": 8,
		"num_key_value_heads": 1,
		"head_dim": 256,
		"rms_norm_eps": 1e-6,
		"rope_theta": 10000.0,
		"vocab_size": 256000,
		"tie_word_embeddings": true,
		"eos_token_id": [1, 107]
	}`)
	a, err := LoadGemmaArgs(cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if a.Arch != "gemma" {
		t.Errorf("arch: got %q want gemma", a.Arch)
	}
	if !a.EmbedScale {
		t.Error("gemma must scale the token embeddings by sqrt(hidden_size)")
	}
	if !a.GeGLU {
		t.Error("gemma must use the GELU MLP gate")
	}
	if !a.NormOnePlus {
		t.Error("gemma must apply its RMSNorm weights as (1 + weight)")
	}
	if a.QKNorm {
		t.Error("gemma must not use per-head query/key norm")
	}
	if a.AttentionBias {
		t.Error("gemma must not bias the attention projections")
	}
	if a.HeadDim != 256 { // independent of hidden_size / num_attention_heads (256)
		t.Errorf("head dim: got %d want 256", a.HeadDim)
	}
	if !reflect.DeepEqual(a.EOSTokenIDs, []int{1, 107}) {
		t.Errorf("eos ids: got %v want [1 107]", a.EOSTokenIDs)
	}
}

func TestLoadGemmaArgsDefaults(t *testing.T) {
	// A terse config still picks up the Gemma rope base and norm epsilon, plus
	// the three structural flags, from the family defaults.
	cfg := []byte(`{
		"model_type": "gemma",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 10
	}`)
	a, err := LoadGemmaArgs(cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if a.RopeTheta != 10000.0 {
		t.Errorf("rope theta default: got %v want 10000", a.RopeTheta)
	}
	if a.RMSNormEps != 1e-6 {
		t.Errorf("rms eps default: got %v want 1e-6", a.RMSNormEps)
	}
	if !a.EmbedScale || !a.GeGLU || !a.NormOnePlus {
		t.Errorf("gemma flags must all be on: scale=%v geglu=%v oneplus=%v",
			a.EmbedScale, a.GeGLU, a.NormOnePlus)
	}
}

func TestLoadArgsDispatchGemma(t *testing.T) {
	cfg := []byte(`{"model_type":"gemma","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":10}`)
	a, err := LoadArgs(cfg)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if a.Arch != "gemma" {
		t.Errorf("arch: got %q want gemma", a.Arch)
	}
	if !a.EmbedScale || !a.GeGLU || !a.NormOnePlus {
		t.Error("dispatched gemma must carry the three structural flags")
	}
}

// TestGemmaFlagsOnlyForGemma guards that the other dense families leave the
// Gemma-specific flags off: none of Qwen3, Qwen2, Llama, or Mistral scales the
// embeddings, uses the GELU gate, or applies the (1 + weight) norm convention.
func TestGemmaFlagsOnlyForGemma(t *testing.T) {
	cases := []struct {
		mt   string
		load func([]byte) (DenseArgs, error)
	}{
		{"qwen3", LoadQwen3Args},
		{"qwen2", LoadQwen2Args},
		{"llama", LoadLlamaArgs},
		{"mistral", LoadLlamaArgs},
	}
	for _, c := range cases {
		cfg := []byte(`{"model_type":"` + c.mt + `","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10}`)
		a, err := c.load(cfg)
		if err != nil {
			t.Fatalf("%s: load: %v", c.mt, err)
		}
		if a.EmbedScale || a.GeGLU || a.NormOnePlus {
			t.Errorf("%s must not carry any Gemma flag: scale=%v geglu=%v oneplus=%v",
				c.mt, a.EmbedScale, a.GeGLU, a.NormOnePlus)
		}
	}
}
