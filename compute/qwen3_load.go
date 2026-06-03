// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"fmt"

	"github.com/tamnd/gomlx/mlxgo"
)

// NewQwen3Model assembles a runnable model from a loaded safetensors file and a
// parsed config. Weight names follow the Hugging Face Qwen3 checkpoint. When the
// config ties the word embeddings, or the checkpoint ships no separate lm_head,
// the embedding matrix doubles as the output projection.
func NewQwen3Model(st *SafeTensors, args Qwen3Args) (*Qwen3Model, error) {
	m := &Qwen3Model{Args: args}

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

	m.Layers = make([]Qwen3Layer, args.NumHiddenLayers)
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
			{&m.Layers[i].QNorm, p + "self_attn.q_norm.weight"},
			{&m.Layers[i].KNorm, p + "self_attn.k_norm.weight"},
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
	}

	m.SetScale()
	return m, nil
}
