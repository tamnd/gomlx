// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package compute

import (
	"testing"

	"github.com/tamnd/gomlx/mlxgo"
)

// fill returns a slice of n values that ramp up and wrap, giving each weight a
// distinct but bounded value so the forward pass produces finite logits.
func fill(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%7)-3) * 0.1
	}
	return out
}

func arr(t *testing.T, shape ...int) mlxgo.Array {
	t.Helper()
	n := 1
	for _, d := range shape {
		n *= d
	}
	a, err := mlxgo.FromFloat32(shape, fill(n))
	if err != nil {
		t.Fatalf("FromFloat32 %v: %v", shape, err)
	}
	return a
}

func ones(t *testing.T, n int) mlxgo.Array {
	t.Helper()
	data := make([]float32, n)
	for i := range data {
		data[i] = 1
	}
	a, err := mlxgo.FromFloat32([]int{n}, data)
	if err != nil {
		t.Fatalf("FromFloat32 ones: %v", err)
	}
	return a
}

// TestQwen3ForwardSmoke builds a tiny Qwen3 model by hand and runs a prompt
// pass followed by one cached decode step on the GPU. It checks shapes and that
// the logits are finite; numeric parity against the reference is a separate
// test that needs a real checkpoint.
func TestQwen3ForwardSmoke(t *testing.T) {
	const (
		vocab  = 8
		hidden = 8
		heads  = 2
		kv     = 1
		hd     = 4
		inter  = 16
		layers = 2
	)
	args := DenseArgs{
		Arch:              "qwen3",
		QKNorm:            true,
		HiddenSize:        hidden,
		IntermediateSize:  inter,
		NumHiddenLayers:   layers,
		NumAttentionHeads: heads,
		NumKeyValueHeads:  kv,
		HeadDim:           hd,
		RMSNormEps:        1e-6,
		RopeTheta:         1000000,
		VocabSize:         vocab,
	}
	m := &DenseModel{Args: args}
	m.Embed = arr(t, vocab, hidden)
	m.Norm = ones(t, hidden)
	m.LMHead = arr(t, vocab, hidden)
	for range layers {
		m.Layers = append(m.Layers, DenseLayer{
			InputNorm:    ones(t, hidden),
			QProj:        arr(t, heads*hd, hidden),
			KProj:        arr(t, kv*hd, hidden),
			VProj:        arr(t, kv*hd, hidden),
			OProj:        arr(t, hidden, heads*hd),
			QNorm:        ones(t, hd),
			KNorm:        ones(t, hd),
			PostAttnNorm: ones(t, hidden),
			Gate:         arr(t, inter, hidden),
			Up:           arr(t, inter, hidden),
			Down:         arr(t, hidden, inter),
		})
	}
	m.SetScale()

	caches := m.NewCaches()

	// Prompt pass: three tokens.
	logits, err := m.Forward([]int32{1, 5, 2}, caches, 0)
	if err != nil {
		t.Fatalf("prompt forward: %v", err)
	}
	if s := logits.Shape(); len(s) != 2 || s[0] != 3 || s[1] != vocab {
		t.Fatalf("prompt logits shape: got %v want [3 %d]", s, vocab)
	}
	checkFinite(t, logits)

	// Decode step: one new token, cache now holds three positions.
	logits2, err := m.Forward([]int32{4}, caches, 3)
	if err != nil {
		t.Fatalf("decode forward: %v", err)
	}
	if s := logits2.Shape(); len(s) != 2 || s[0] != 1 || s[1] != vocab {
		t.Fatalf("decode logits shape: got %v want [1 %d]", s, vocab)
	}
	checkFinite(t, logits2)

	// The KV cache should now span four positions on each layer.
	for i, c := range caches {
		if !c.Valid {
			t.Fatalf("layer %d cache invalid", i)
		}
		if s := c.K.Shape(); s[2] != 4 {
			t.Errorf("layer %d cache length: got %v want seq 4", i, s)
		}
	}
}

func checkFinite(t *testing.T, a mlxgo.Array) {
	t.Helper()
	vals, err := a.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32: %v", err)
	}
	for i, v := range vals {
		if v != v || v > 1e30 || v < -1e30 {
			t.Fatalf("logit %d not finite: %v", i, v)
		}
	}
}
