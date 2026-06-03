// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package compute

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"testing"
)

// stBuilder serializes tensors into the safetensors wire format so a model can
// be assembled and run without a checkpoint on disk.
type stBuilder struct {
	order   []string
	entries map[string]headerEntry
	data    []byte
}

func newSTBuilder() *stBuilder {
	return &stBuilder{entries: map[string]headerEntry{}}
}

func (b *stBuilder) add(name string, shape []int, vals []float32) {
	begin := int64(len(b.data))
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	b.data = append(b.data, buf...)
	b.order = append(b.order, name)
	b.entries[name] = headerEntry{Dtype: "F32", Shape: shape, DataOffsets: []int64{begin, int64(len(b.data))}}
}

func (b *stBuilder) ramp(name string, shape ...int) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32((i%7)-3) * 0.1
	}
	b.add(name, shape, vals)
}

func (b *stBuilder) onesT(name string, n int) {
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = 1
	}
	b.add(name, []int{n}, vals)
}

func (b *stBuilder) bytes(t *testing.T) []byte {
	t.Helper()
	// Header JSON must preserve insertion order for offsets, but parsing does not
	// depend on order, so a plain map is fine here.
	obj := map[string]headerEntry{}
	for _, name := range b.order {
		obj[name] = b.entries[name]
	}
	hdr, err := json.Marshal(obj)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	out := make([]byte, 8+len(hdr)+len(b.data))
	binary.LittleEndian.PutUint64(out, uint64(len(hdr)))
	copy(out[8:], hdr)
	copy(out[8+len(hdr):], b.data)
	return out
}

// TestNewQwen3ModelFromBytes serializes a tiny tied-embedding checkpoint,
// assembles the model through the same path a real load takes, and runs a
// forward pass on the GPU.
func TestNewQwen3ModelFromBytes(t *testing.T) {
	const (
		vocab  = 8
		hidden = 8
		heads  = 2
		kv     = 1
		hd     = 4
		inter  = 16
		layers = 2
	)
	args := Qwen3Args{
		HiddenSize:        hidden,
		IntermediateSize:  inter,
		NumHiddenLayers:   layers,
		NumAttentionHeads: heads,
		NumKeyValueHeads:  kv,
		HeadDim:           hd,
		RMSNormEps:        1e-6,
		RopeTheta:         1000000,
		VocabSize:         vocab,
		TieWordEmbeddings: true,
	}

	b := newSTBuilder()
	b.ramp("model.embed_tokens.weight", vocab, hidden)
	b.onesT("model.norm.weight", hidden)
	for i := range layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		b.onesT(p+"input_layernorm.weight", hidden)
		b.ramp(p+"self_attn.q_proj.weight", heads*hd, hidden)
		b.ramp(p+"self_attn.k_proj.weight", kv*hd, hidden)
		b.ramp(p+"self_attn.v_proj.weight", kv*hd, hidden)
		b.ramp(p+"self_attn.o_proj.weight", hidden, heads*hd)
		b.onesT(p+"self_attn.q_norm.weight", hd)
		b.onesT(p+"self_attn.k_norm.weight", hd)
		b.onesT(p+"post_attention_layernorm.weight", hidden)
		b.ramp(p+"mlp.gate_proj.weight", inter, hidden)
		b.ramp(p+"mlp.up_proj.weight", inter, hidden)
		b.ramp(p+"mlp.down_proj.weight", hidden, inter)
	}

	st, err := FromBytes(b.bytes(t))
	if err != nil {
		t.Fatalf("FromBytes: %v", err)
	}
	defer st.Close()

	m, err := NewQwen3Model(st, args)
	if err != nil {
		t.Fatalf("NewQwen3Model: %v", err)
	}
	if len(m.Layers) != layers {
		t.Fatalf("layers: got %d want %d", len(m.Layers), layers)
	}

	logits, err := m.Forward([]int32{1, 5, 2}, m.NewCaches(), 0)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if s := logits.Shape(); len(s) != 2 || s[0] != 3 || s[1] != vocab {
		t.Fatalf("logits shape: got %v want [3 %d]", s, vocab)
	}
	checkFinite(t, logits)
}
