// SPDX-License-Identifier: Apache-2.0

package compute

import "testing"

func TestLoadQwen3ArgsFull(t *testing.T) {
	cfg := []byte(`{
		"model_type": "qwen3",
		"hidden_size": 1024,
		"intermediate_size": 3072,
		"num_hidden_layers": 28,
		"num_attention_heads": 16,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"rms_norm_eps": 1e-5,
		"rope_theta": 5000000.0,
		"vocab_size": 151936,
		"max_position_embeddings": 40960,
		"tie_word_embeddings": true
	}`)
	a, err := LoadQwen3Args(cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if a.NumKeyValueHeads != 8 || a.HeadDim != 128 {
		t.Errorf("kv heads/head dim: %+v", a)
	}
	if a.RMSNormEps != 1e-5 || a.RopeTheta != 5000000.0 {
		t.Errorf("eps/theta: %+v", a)
	}
	if !a.TieWordEmbeddings {
		t.Error("tie_word_embeddings should be true")
	}
}

func TestLoadQwen3ArgsDefaults(t *testing.T) {
	// Minimal config: kv heads, head dim, eps, and theta come from defaults.
	cfg := []byte(`{
		"hidden_size": 2048,
		"num_hidden_layers": 24,
		"num_attention_heads": 16,
		"vocab_size": 151936
	}`)
	a, err := LoadQwen3Args(cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if a.NumKeyValueHeads != 16 {
		t.Errorf("kv heads default should equal attention heads, got %d", a.NumKeyValueHeads)
	}
	if a.HeadDim != 128 { // 2048 / 16
		t.Errorf("head dim default: got %d want 128", a.HeadDim)
	}
	if a.RMSNormEps != 1e-6 {
		t.Errorf("eps default: got %v", a.RMSNormEps)
	}
	if a.RopeTheta != 1000000.0 {
		t.Errorf("theta default: got %v", a.RopeTheta)
	}
}

func TestLoadQwen3ArgsRejectsBad(t *testing.T) {
	cases := map[string][]byte{
		"missing hidden_size": []byte(`{"num_hidden_layers":1,"num_attention_heads":1,"vocab_size":10}`),
		"zero layers":         []byte(`{"hidden_size":8,"num_attention_heads":1,"vocab_size":10}`),
		"indivisible heads":   []byte(`{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":3,"num_key_value_heads":2,"vocab_size":10}`),
		"not json":            []byte(`{`),
	}
	for name, cfg := range cases {
		if _, err := LoadQwen3Args(cfg); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}
