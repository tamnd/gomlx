// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package compute

import (
	"testing"
)

// TestGemmaForwardSmoke builds a tiny Gemma model by hand and runs a prompt pass
// followed by one cached decode step on the GPU. It turns on the three Gemma
// flags so the forward exercises the parts that are unique to the family: the
// sqrt(hidden_size) embedding scale, the exact-GELU MLP gate, and the (1 + weight)
// RMSNorm convention. As with the Qwen3 smoke test this only checks shapes and
// that the logits are finite; numeric parity against the reference needs a real
// checkpoint.
func TestGemmaForwardSmoke(t *testing.T) {
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
		Arch:              "gemma",
		HiddenSize:        hidden,
		IntermediateSize:  inter,
		NumHiddenLayers:   layers,
		NumAttentionHeads: heads,
		NumKeyValueHeads:  kv,
		HeadDim:           hd,
		RMSNormEps:        1e-6,
		RopeTheta:         10000,
		VocabSize:         vocab,
		EmbedScale:        true,
		GeGLU:             true,
		NormOnePlus:       true,
	}
	m := &DenseModel{Args: args}
	m.Embed = arr(t, vocab, hidden)
	m.LMHead = m.Embed // Gemma ties the embeddings.

	// Fold the (1 + weight) convention into the norm weights, exactly as the real
	// loader does, so the hand-built model matches the production forward.
	var err error
	if m.Norm, err = onePlus(ones(t, hidden)); err != nil {
		t.Fatalf("fold final norm: %v", err)
	}
	for range layers {
		in, ferr := onePlus(ones(t, hidden))
		if ferr != nil {
			t.Fatalf("fold input norm: %v", ferr)
		}
		post, ferr := onePlus(ones(t, hidden))
		if ferr != nil {
			t.Fatalf("fold post-attn norm: %v", ferr)
		}
		m.Layers = append(m.Layers, DenseLayer{
			InputNorm:    in,
			QProj:        arr(t, heads*hd, hidden),
			KProj:        arr(t, kv*hd, hidden),
			VProj:        arr(t, kv*hd, hidden),
			OProj:        arr(t, hidden, heads*hd),
			PostAttnNorm: post,
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
