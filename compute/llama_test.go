// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"reflect"
	"testing"
)

func TestLoadLlamaArgs(t *testing.T) {
	cfg := []byte(`{
		"model_type": "llama",
		"hidden_size": 2048,
		"intermediate_size": 5632,
		"num_hidden_layers": 22,
		"num_attention_heads": 32,
		"num_key_value_heads": 4,
		"rms_norm_eps": 1e-5,
		"rope_theta": 500000.0,
		"vocab_size": 128256,
		"tie_word_embeddings": false,
		"eos_token_id": [128001, 128009]
	}`)
	a, err := LoadLlamaArgs(cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if a.Arch != "llama" {
		t.Errorf("arch: got %q want llama", a.Arch)
	}
	if a.QKNorm {
		t.Error("llama must not use per-head query/key norm")
	}
	if a.NumKeyValueHeads != 4 {
		t.Errorf("kv heads: got %d want 4", a.NumKeyValueHeads)
	}
	if a.HeadDim != 64 { // 2048 / 32
		t.Errorf("head dim default: got %d want 64", a.HeadDim)
	}
	if a.RopeTheta != 500000.0 {
		t.Errorf("rope theta: got %v", a.RopeTheta)
	}
	if !reflect.DeepEqual(a.EOSTokenIDs, []int{128001, 128009}) {
		t.Errorf("eos ids: got %v", a.EOSTokenIDs)
	}
}

func TestLoadLlamaArgsMistralDefaults(t *testing.T) {
	// A Mistral config that omits rope_theta should pick up the Mistral default,
	// not the Llama one.
	cfg := []byte(`{
		"model_type": "mistral",
		"hidden_size": 4096,
		"num_hidden_layers": 32,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"sliding_window": 4096,
		"vocab_size": 32000
	}`)
	a, err := LoadLlamaArgs(cfg)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if a.Arch != "mistral" {
		t.Errorf("arch: got %q want mistral", a.Arch)
	}
	if a.RopeTheta != 1000000.0 {
		t.Errorf("mistral rope default: got %v want 1e6", a.RopeTheta)
	}
	if a.SlidingWindow != 4096 {
		t.Errorf("sliding window: got %d", a.SlidingWindow)
	}
	if a.QKNorm {
		t.Error("mistral must not use per-head query/key norm")
	}
}

func TestLoadArgsDispatch(t *testing.T) {
	cases := []struct {
		name   string
		cfg    string
		arch   string
		qkNorm bool
		ok     bool
	}{
		{
			name:   "qwen3",
			cfg:    `{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10}`,
			arch:   "qwen3",
			qkNorm: true,
			ok:     true,
		},
		{
			name:   "llama",
			cfg:    `{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10}`,
			arch:   "llama",
			qkNorm: false,
			ok:     true,
		},
		{
			name:   "mistral",
			cfg:    `{"model_type":"mistral","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10}`,
			arch:   "mistral",
			qkNorm: false,
			ok:     true,
		},
		{
			name: "unsupported type",
			cfg:  `{"model_type":"gemma","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":10}`,
			ok:   false,
		},
		{
			name: "missing type",
			cfg:  `{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"vocab_size":10}`,
			ok:   false,
		},
	}
	for _, c := range cases {
		a, err := LoadArgs([]byte(c.cfg))
		if c.ok != (err == nil) {
			t.Fatalf("%s: ok=%v but err=%v", c.name, c.ok, err)
		}
		if !c.ok {
			continue
		}
		if a.Arch != c.arch || a.QKNorm != c.qkNorm {
			t.Errorf("%s: got arch=%q qkNorm=%v", c.name, a.Arch, a.QKNorm)
		}
	}
}

func TestParseEOS(t *testing.T) {
	// A scalar eos_token_id is the common Qwen3/Mistral form; a list is the Llama3
	// form; an absent field yields nil so the caller falls back to a default.
	scalar := []byte(`{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10,"eos_token_id":151645}`)
	a, err := LoadArgs(scalar)
	if err != nil {
		t.Fatalf("scalar: %v", err)
	}
	if !reflect.DeepEqual(a.EOSTokenIDs, []int{151645}) {
		t.Errorf("scalar eos: got %v", a.EOSTokenIDs)
	}

	absent := []byte(`{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":10}`)
	if a, _ := LoadArgs(absent); a.EOSTokenIDs != nil {
		t.Errorf("absent eos should be nil, got %v", a.EOSTokenIDs)
	}
}
