// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"fmt"
	"math"

	"github.com/tamnd/gomlx/mlxgo"
)

// DenseLayer holds the weights of one transformer block. Names follow the
// Hugging Face checkpoint: the four attention projections, the optional per-head
// query and key norms (Qwen3 only), the two layer norms, and the three MLP
// projections. All projection weights are stored as the checkpoint stores them,
// shaped [out_features, in_features]. QNorm and KNorm are left as zero-value
// arrays for architectures that do not use them.
type DenseLayer struct {
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

// DenseModel is a loaded dense decoder (Qwen3, Llama, or Mistral) ready to run
// forward passes. The families share this one implementation; DenseArgs.QKNorm
// selects the only structural difference in the attention.
type DenseModel struct {
	Args   DenseArgs
	Embed  mlxgo.Array // [vocab, hidden]
	Layers []DenseLayer
	Norm   mlxgo.Array // final norm [hidden]
	LMHead mlxgo.Array // [vocab, hidden]; equals Embed when weights are tied
	scale  float32     // attention scale 1/sqrt(head_dim)
}

// newDenseModel assembles a runnable model from a loaded safetensors file and a
// parsed config. Weight names follow the Hugging Face checkpoint layout, which is
// identical across these families. The per-head query/key norms are loaded only
// when the architecture uses them. When the config ties the word embeddings, or
// the checkpoint ships no separate lm_head, the embedding matrix doubles as the
// output projection.
func newDenseModel(st *SafeTensors, args DenseArgs) (*DenseModel, error) {
	m := &DenseModel{Args: args}

	var err error
	if m.Embed, err = LoadArray(st, "model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	if m.Norm, err = LoadArray(st, "model.norm.weight"); err != nil {
		return nil, err
	}

	if args.TieWordEmbeddings || !st.Has("lm_head.weight") {
		m.LMHead = m.Embed
	} else if m.LMHead, err = LoadArray(st, "lm_head.weight"); err != nil {
		return nil, err
	}

	m.Layers = make([]DenseLayer, args.NumHiddenLayers)
	for i := range m.Layers {
		p := fmt.Sprintf("model.layers.%d.", i)
		fields := []struct {
			dst  *mlxgo.Array
			name string
		}{
			{&m.Layers[i].InputNorm, p + "input_layernorm.weight"},
			{&m.Layers[i].QProj, p + "self_attn.q_proj.weight"},
			{&m.Layers[i].KProj, p + "self_attn.k_proj.weight"},
			{&m.Layers[i].VProj, p + "self_attn.v_proj.weight"},
			{&m.Layers[i].OProj, p + "self_attn.o_proj.weight"},
			{&m.Layers[i].PostAttnNorm, p + "post_attention_layernorm.weight"},
			{&m.Layers[i].Gate, p + "mlp.gate_proj.weight"},
			{&m.Layers[i].Up, p + "mlp.up_proj.weight"},
			{&m.Layers[i].Down, p + "mlp.down_proj.weight"},
		}
		for _, f := range fields {
			if *f.dst, err = LoadArray(st, f.name); err != nil {
				return nil, err
			}
		}
		if args.QKNorm {
			if m.Layers[i].QNorm, err = LoadArray(st, p+"self_attn.q_norm.weight"); err != nil {
				return nil, err
			}
			if m.Layers[i].KNorm, err = LoadArray(st, p+"self_attn.k_norm.weight"); err != nil {
				return nil, err
			}
		}
	}

	m.SetScale()
	return m, nil
}

// NewModel builds a runnable model from weights and parsed args, for any
// supported dense architecture. The arch-specific entry points (NewQwen3Model,
// NewLlamaModel) and this generic one all share the same loader; which weights
// are read follows from args.QKNorm.
func NewModel(st *SafeTensors, args DenseArgs) (*DenseModel, error) {
	return newDenseModel(st, args)
}

// LayerCache holds the running key and value tensors for one layer across decode
// steps. Both are shaped [1, n_kv_heads, seq, head_dim].
type LayerCache struct {
	K     mlxgo.Array
	V     mlxgo.Array
	Valid bool
}

// NewCaches returns an empty cache per layer.
func (m *DenseModel) NewCaches() []LayerCache {
	return make([]LayerCache, len(m.Layers))
}

// SetScale fills the derived attention scale. newDenseModel calls it; it is
// exported only so a hand-built test model can set it too.
func (m *DenseModel) SetScale() {
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
func (m *DenseModel) Forward(tokens []int32, caches []LayerCache, offset int) (mlxgo.Array, error) {
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
func (m *DenseModel) block(h mlxgo.Array, l *DenseLayer, c *LayerCache, seq, offset int) (mlxgo.Array, error) {
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

	// Qwen3 applies a per-head RMSNorm to the query and key over the head
	// dimension; Llama and Mistral skip it.
	if a.QKNorm {
		if q, err = mlxgo.RMSNorm(q, l.QNorm, eps); err != nil {
			return mlxgo.Array{}, err
		}
		if k, err = mlxgo.RMSNorm(k, l.KNorm, eps); err != nil {
			return mlxgo.Array{}, err
		}
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
