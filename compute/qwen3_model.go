// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"fmt"
	"math"

	"github.com/tamnd/gomlx/mlxgo"
)

// Qwen3Layer holds the weights of one transformer block. Names follow the
// Hugging Face checkpoint: the four attention projections, the Qwen3 per-head
// query and key norms, the two layer norms, and the three MLP projections. All
// projection weights are stored as the checkpoint stores them, shaped
// [out_features, in_features].
type Qwen3Layer struct {
	InputNorm    mlxgo.Array
	QProj        mlxgo.Array
	KProj        mlxgo.Array
	VProj        mlxgo.Array
	OProj        mlxgo.Array
	QNorm        mlxgo.Array
	KNorm        mlxgo.Array
	PostAttnNorm mlxgo.Array
	Gate         mlxgo.Array
	Up           mlxgo.Array
	Down         mlxgo.Array
}

// Qwen3Model is a loaded Qwen3 dense model ready to run forward passes.
type Qwen3Model struct {
	Args   Qwen3Args
	Embed  mlxgo.Array // [vocab, hidden]
	Layers []Qwen3Layer
	Norm   mlxgo.Array // final norm [hidden]
	LMHead mlxgo.Array // [vocab, hidden]; equals Embed when weights are tied
	scale  float32     // attention scale 1/sqrt(head_dim)
}

// LayerCache holds the running key and value tensors for one layer across
// decode steps. Both are shaped [1, n_kv_heads, seq, head_dim].
type LayerCache struct {
	K     mlxgo.Array
	V     mlxgo.Array
	Valid bool
}

// NewCaches returns an empty cache per layer.
func (m *Qwen3Model) NewCaches() []LayerCache {
	return make([]LayerCache, len(m.Layers))
}

// SetScale fills the derived attention scale. NewQwen3Model calls it; it is
// exported only so a hand-built test model can set it too.
func (m *Qwen3Model) SetScale() {
	m.scale = float32(1.0 / math.Sqrt(float64(m.Args.HeadDim)))
}

// linear computes x @ W^T for a checkpoint weight stored as [out, in].
func linear(x, w mlxgo.Array) (mlxgo.Array, error) {
	wt, err := mlxgo.Transpose(w, []int{1, 0})
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.MatMul(x, wt)
}

// Forward runs the model over tokens, appending to the per-layer caches, and
// returns the logits for every position, shaped [seq, vocab], as float32.
// offset is the number of positions already in the cache, which sets the RoPE
// phase for the new tokens.
func (m *Qwen3Model) Forward(tokens []int32, caches []LayerCache, offset int) (mlxgo.Array, error) {
	a := m.Args
	seq := len(tokens)

	raw := make([]byte, 4*seq)
	for i, t := range tokens {
		u := uint32(t)
		raw[i*4] = byte(u)
		raw[i*4+1] = byte(u >> 8)
		raw[i*4+2] = byte(u >> 16)
		raw[i*4+3] = byte(u >> 24)
	}
	ids, err := mlxgo.FromRawBytes([]int{seq}, mlxgo.I32, raw)
	if err != nil {
		return mlxgo.Array{}, err
	}

	h, err := mlxgo.Take(m.Embed, ids) // [seq, hidden]
	if err != nil {
		return mlxgo.Array{}, fmt.Errorf("embed: %w", err)
	}

	for i := range m.Layers {
		h, err = m.block(h, &m.Layers[i], &caches[i], seq, offset)
		if err != nil {
			return mlxgo.Array{}, fmt.Errorf("layer %d: %w", i, err)
		}
	}

	h, err = mlxgo.RMSNorm(h, m.Norm, float32(a.RMSNormEps))
	if err != nil {
		return mlxgo.Array{}, fmt.Errorf("final norm: %w", err)
	}
	logits, err := linear(h, m.LMHead) // [seq, vocab]
	if err != nil {
		return mlxgo.Array{}, fmt.Errorf("lm head: %w", err)
	}
	return mlxgo.Astype(logits, mlxgo.F32)
}

// block runs one transformer layer.
func (m *Qwen3Model) block(h mlxgo.Array, l *Qwen3Layer, c *LayerCache, seq, offset int) (mlxgo.Array, error) {
	a := m.Args
	eps := float32(a.RMSNormEps)

	// Attention.
	hn, err := mlxgo.RMSNorm(h, l.InputNorm, eps)
	if err != nil {
		return mlxgo.Array{}, err
	}
	q, err := linear(hn, l.QProj)
	if err != nil {
		return mlxgo.Array{}, err
	}
	k, err := linear(hn, l.KProj)
	if err != nil {
		return mlxgo.Array{}, err
	}
	v, err := linear(hn, l.VProj)
	if err != nil {
		return mlxgo.Array{}, err
	}

	// Split heads: [seq, H*D] -> [1, H, seq, D].
	q, err = toHeads(q, seq, a.NumAttentionHeads, a.HeadDim)
	if err != nil {
		return mlxgo.Array{}, err
	}
	k, err = toHeads(k, seq, a.NumKeyValueHeads, a.HeadDim)
	if err != nil {
		return mlxgo.Array{}, err
	}
	v, err = toHeads(v, seq, a.NumKeyValueHeads, a.HeadDim)
	if err != nil {
		return mlxgo.Array{}, err
	}

	// Qwen3 per-head QK norm over the head dimension.
	if q, err = mlxgo.RMSNorm(q, l.QNorm, eps); err != nil {
		return mlxgo.Array{}, err
	}
	if k, err = mlxgo.RMSNorm(k, l.KNorm, eps); err != nil {
		return mlxgo.Array{}, err
	}

	// Rotary embeddings phased by the cache offset.
	base := float32(a.RopeTheta)
	if q, err = mlxgo.RoPE(q, a.HeadDim, false, base, 1, offset); err != nil {
		return mlxgo.Array{}, err
	}
	if k, err = mlxgo.RoPE(k, a.HeadDim, false, base, 1, offset); err != nil {
		return mlxgo.Array{}, err
	}

	// Append to the KV cache.
	if c.Valid {
		if k, err = mlxgo.Concat([]mlxgo.Array{c.K, k}, 2); err != nil {
			return mlxgo.Array{}, err
		}
		if v, err = mlxgo.Concat([]mlxgo.Array{c.V, v}, 2); err != nil {
			return mlxgo.Array{}, err
		}
	}
	c.K, c.V, c.Valid = k, v, true

	attn, err := mlxgo.SDPA(q, k, v, m.scale, true)
	if err != nil {
		return mlxgo.Array{}, err
	}

	// Merge heads: [1, H, seq, D] -> [seq, H*D].
	attn, err = fromHeads(attn, seq, a.NumAttentionHeads, a.HeadDim)
	if err != nil {
		return mlxgo.Array{}, err
	}
	o, err := linear(attn, l.OProj)
	if err != nil {
		return mlxgo.Array{}, err
	}
	if h, err = mlxgo.Add(h, o); err != nil {
		return mlxgo.Array{}, err
	}

	// MLP (SwiGLU).
	hn2, err := mlxgo.RMSNorm(h, l.PostAttnNorm, eps)
	if err != nil {
		return mlxgo.Array{}, err
	}
	gate, err := linear(hn2, l.Gate)
	if err != nil {
		return mlxgo.Array{}, err
	}
	gate, err = mlxgo.Silu(gate)
	if err != nil {
		return mlxgo.Array{}, err
	}
	up, err := linear(hn2, l.Up)
	if err != nil {
		return mlxgo.Array{}, err
	}
	act, err := mlxgo.Multiply(gate, up)
	if err != nil {
		return mlxgo.Array{}, err
	}
	down, err := linear(act, l.Down)
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.Add(h, down)
}

// toHeads reshapes [seq, heads*dim] to [1, heads, seq, dim].
func toHeads(x mlxgo.Array, seq, heads, dim int) (mlxgo.Array, error) {
	r, err := mlxgo.Reshape(x, []int{1, seq, heads, dim})
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.Transpose(r, []int{0, 2, 1, 3})
}

// fromHeads reshapes [1, heads, seq, dim] back to [seq, heads*dim].
func fromHeads(x mlxgo.Array, seq, heads, dim int) (mlxgo.Array, error) {
	t, err := mlxgo.Transpose(x, []int{0, 2, 1, 3})
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.Reshape(t, []int{seq, heads * dim})
}
