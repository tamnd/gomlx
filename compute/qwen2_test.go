// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"reflect"
	"testing"
)

func TestLoadQwen2Args(t *testing.T) {
	cfg := []byte(`{
		"model_type": "qwen2",
		"hidden_size": 1536,
		"intermediate_size": 8960,
		"num_hidden_layers": 28,
		"num_attention_heads": 12,
		"num_key_value_heads": 2,
		"rms_norm_eps": 1e-6,
		"rope_theta": 1000000.0,
		"vocab_size": 151936,
		"tie_word_embeddings": true,
		"eos_token_id": 151645
	}`)
	a, err := LoadQwen2Args(cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if a.Arch != "qwen2" {
		t.Errorf("arch: got %q want qwen2", a.Arch)
	}
	if a.QKNorm {
		t.Error("qwen2 must not use per-head query/key norm")
	}
	if !a.AttentionBias {
		t.Error("qwen2 must bias the query/key/value projections")
	}
	if !a.TieWordEmbeddings {
		t.Error("this qwen2 config ties the word embeddings")
	}
	if a.NumKeyValueHeads != 2 {
		t.Errorf("kv heads: got %d want 2", a.NumKeyValueHeads)
	}
	if a.HeadDim != 128 { // 1536 / 12
		t.Errorf("head dim default: got %d want 128", a.HeadDim)
	}
	if !reflect.DeepEqual(a.EOSTokenIDs, []int{151645}) {
		t.Errorf("eos ids: got %v", a.EOSTokenIDs)
	}
}

func TestLoadQwen2ArgsForcesBias(t *testing.T) {
	// The Qwen2 architecture biases its attention projections regardless of
	// whether the config spells attention_bias out, so a config that omits the
	// field must still come back with the bias enabled.
	cfg := []byte(`{
		"model_type": "qwen2",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"vocab_size": 10
	}`)
	a, err := LoadQwen2Args(cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !a.AttentionBias {
		t.Error("qwen2 must force attention bias on even when the config omits it")
	}
	// The Qwen2 rope and norm defaults should apply for a terse config.
	if a.RopeTheta != 1000000.0 {
		t.Errorf("rope theta default: got %v want 1e6", a.RopeTheta)
	}
	if a.RMSNormEps != 1e-6 {
		t.Errorf("rms eps default: got %v want 1e-6", a.RMSNormEps)
	}
}

func TestLoadArgsDispatchQwen2(t *testing.T) {
	cfg := []byte(`{"model_type":"qwen2","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10}`)
	a, err := LoadArgs(cfg)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if a.Arch != "qwen2" {
		t.Errorf("arch: got %q want qwen2", a.Arch)
	}
	if !a.AttentionBias {
		t.Error("dispatched qwen2 must carry the attention bias")
	}
	if a.QKNorm {
		t.Error("qwen2 must not use per-head query/key norm")
	}
}

// TestQwen2BiasOnlyForQwen2 guards the default for the other families: none of
// Qwen3, Llama, or Mistral biases the attention projections, so a config without
// attention_bias must leave the flag off for them.
func TestQwen2BiasOnlyForQwen2(t *testing.T) {
	cases := []struct {
		mt   string
		load func([]byte) (DenseArgs, error)
	}{
		{"qwen3", LoadQwen3Args},
		{"llama", LoadLlamaArgs},
		{"mistral", LoadLlamaArgs},
	}
	for _, c := range cases {
		cfg := []byte(`{"model_type":"` + c.mt + `","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10}`)
		a, err := c.load(cfg)
		if err != nil {
			t.Fatalf("%s: load: %v", c.mt, err)
		}
		if a.AttentionBias {
			t.Errorf("%s must not bias the attention projections by default", c.mt)
		}
	}
}
