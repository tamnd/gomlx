// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"fmt"

	"github.com/tamnd/gomlx/mlxgo"
)

// maskNeg is the additive value for a masked attention position. It is large
// enough to zero the softmax weight without overflowing in bf16/f16.
const maskNeg = -1e9

// BatchCache holds the running keys and values for one layer across a batch of
// sequences decoding in lockstep. Both are shaped [batch, n_kv_heads, len,
// head_dim]. Every sequence in the batch shares the same cache length; prompts
// of different lengths are left-padded so their real tokens end at the same
// absolute position, which keeps a single rotary offset valid for the whole
// batch.
type BatchCache struct {
	K     mlxgo.Array
	V     mlxgo.Array
	Valid bool
}

// Batch is a set of sequences generating together. The model runs one forward
// pass per decode step over the whole batch, which is what turns the big
// projection and MLP matmuls into batched matmuls and lifts throughput far
// above serialized single-stream decode.
type Batch struct {
	m      *Qwen3Model
	caches []BatchCache

	n    int   // number of sequences
	pads []int // left padding per sequence (P - prompt length)
	pos  int   // current cache length L

	scaleArr mlxgo.Array // attention scale as a scalar in the model dtype
}

// NewBatch prepares an empty batch state for n sequences.
func (m *Qwen3Model) NewBatch(n int) *Batch {
	return &Batch{m: m, caches: make([]BatchCache, len(m.Layers)), n: n}
}

// Size reports the number of sequences in the batch.
func (b *Batch) Size() int { return b.n }

// Prefill runs the prompts through the model and returns the logits at each
// sequence's final prompt position, shaped [n, vocab] flattened row-major. The
// prompts may have different lengths; they are left-padded to a common length
// so a shared rotary offset and padded cache stay valid.
func (b *Batch) Prefill(prompts [][]int32) ([]float32, error) {
	b.n = len(prompts)

	P := 0
	for _, p := range prompts {
		if len(p) > P {
			P = len(p)
		}
	}
	if P == 0 {
		return nil, fmt.Errorf("compute: empty prompt batch")
	}
	b.pads = make([]int, b.n)
	for i, p := range prompts {
		b.pads[i] = P - len(p)
	}

	if err := b.initScale(); err != nil {
		return nil, err
	}

	// Build the left-padded token matrix [n, P] as a flat int32 buffer.
	ids := make([]int32, b.n*P)
	for i, p := range prompts {
		copy(ids[i*P+b.pads[i]:], p)
	}
	h, err := b.embed(ids, b.n, P)
	if err != nil {
		return nil, err
	}

	mask, err := b.prefillMask(P)
	if err != nil {
		return nil, err
	}

	for i := range b.m.Layers {
		h, err = b.block(h, &b.m.Layers[i], &b.caches[i], P, 0, mask)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}
	b.pos = P
	return b.lastLogits(h, P)
}

// Decode advances every sequence by one token and returns the next-token
// logits, shaped [n, vocab] flattened row-major. tokens holds the freshly
// sampled token for each sequence.
func (b *Batch) Decode(tokens []int32) ([]float32, error) {
	h, err := b.embed(tokens, b.n, 1)
	if err != nil {
		return nil, err
	}
	offset := b.pos
	mask, err := b.decodeMask(b.pos + 1)
	if err != nil {
		return nil, err
	}
	for i := range b.m.Layers {
		h, err = b.block(h, &b.m.Layers[i], &b.caches[i], 1, offset, mask)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}
	b.pos++
	return b.lastLogits(h, 1)
}

// embed gathers token embeddings for an [n, q] id matrix into [n, q, hidden].
func (b *Batch) embed(ids []int32, n, q int) (mlxgo.Array, error) {
	raw := make([]byte, 4*len(ids))
	for i, t := range ids {
		u := uint32(t)
		raw[i*4] = byte(u)
		raw[i*4+1] = byte(u >> 8)
		raw[i*4+2] = byte(u >> 16)
		raw[i*4+3] = byte(u >> 24)
	}
	idArr, err := mlxgo.FromRawBytes([]int{n * q}, mlxgo.I32, raw)
	if err != nil {
		return mlxgo.Array{}, err
	}
	h, err := mlxgo.Take(b.m.Embed, idArr)
	if err != nil {
		return mlxgo.Array{}, fmt.Errorf("embed: %w", err)
	}
	return mlxgo.Reshape(h, []int{n, q, b.m.Args.HiddenSize})
}

// block runs one transformer layer over a batch. q is the query length (the
// padded prompt length on prefill, 1 on decode); offset is the rotary phase for
// the new tokens; mask is the additive attention mask.
func (b *Batch) block(h mlxgo.Array, l *Qwen3Layer, c *BatchCache, q, offset int, mask mlxgo.Array) (mlxgo.Array, error) {
	a := b.m.Args
	eps := float32(a.RMSNormEps)
	n := b.n

	hn, err := mlxgo.RMSNorm(h, l.InputNorm, eps)
	if err != nil {
		return mlxgo.Array{}, err
	}
	qp, err := linear(hn, l.QProj)
	if err != nil {
		return mlxgo.Array{}, err
	}
	kp, err := linear(hn, l.KProj)
	if err != nil {
		return mlxgo.Array{}, err
	}
	vp, err := linear(hn, l.VProj)
	if err != nil {
		return mlxgo.Array{}, err
	}

	if qp, err = toHeadsB(qp, n, q, a.NumAttentionHeads, a.HeadDim); err != nil {
		return mlxgo.Array{}, err
	}
	if kp, err = toHeadsB(kp, n, q, a.NumKeyValueHeads, a.HeadDim); err != nil {
		return mlxgo.Array{}, err
	}
	if vp, err = toHeadsB(vp, n, q, a.NumKeyValueHeads, a.HeadDim); err != nil {
		return mlxgo.Array{}, err
	}

	if qp, err = mlxgo.RMSNorm(qp, l.QNorm, eps); err != nil {
		return mlxgo.Array{}, err
	}
	if kp, err = mlxgo.RMSNorm(kp, l.KNorm, eps); err != nil {
		return mlxgo.Array{}, err
	}

	base := float32(a.RopeTheta)
	if qp, err = mlxgo.RoPE(qp, a.HeadDim, false, base, 1, offset); err != nil {
		return mlxgo.Array{}, err
	}
	if kp, err = mlxgo.RoPE(kp, a.HeadDim, false, base, 1, offset); err != nil {
		return mlxgo.Array{}, err
	}

	if c.Valid {
		if kp, err = mlxgo.Concat([]mlxgo.Array{c.K, kp}, 2); err != nil {
			return mlxgo.Array{}, err
		}
		if vp, err = mlxgo.Concat([]mlxgo.Array{c.V, vp}, 2); err != nil {
			return mlxgo.Array{}, err
		}
	}
	c.K, c.V, c.Valid = kp, vp, true

	kl := offset + q
	attn, err := b.attention(qp, kp, vp, mask, q, kl)
	if err != nil {
		return mlxgo.Array{}, err
	}
	attn, err = fromHeadsB(attn, n, q, a.NumAttentionHeads, a.HeadDim)
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

	hn2, err := mlxgo.RMSNorm(h, l.PostAttnNorm, eps)
	if err != nil {
		return mlxgo.Array{}, err
	}
	gate, err := linear(hn2, l.Gate)
	if err != nil {
		return mlxgo.Array{}, err
	}
	if gate, err = mlxgo.Silu(gate); err != nil {
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

// lastLogits takes the final query position of each sequence, runs the final
// norm and the LM head over just those rows, and returns [n, vocab] as float32.
func (b *Batch) lastLogits(h mlxgo.Array, q int) ([]float32, error) {
	// h is [n, q, hidden]; the last query position is real for every sequence
	// because prompts are right-aligned within the padded window.
	last, err := sliceLastQ(h, b.n, q, b.m.Args.HiddenSize)
	if err != nil {
		return nil, err
	}
	last, err = mlxgo.RMSNorm(last, b.m.Norm, float32(b.m.Args.RMSNormEps))
	if err != nil {
		return nil, fmt.Errorf("final norm: %w", err)
	}
	logits, err := linear(last, b.m.LMHead)
	if err != nil {
		return nil, fmt.Errorf("lm head: %w", err)
	}
	logits, err = mlxgo.Astype(logits, mlxgo.F32)
	if err != nil {
		return nil, err
	}
	return logits.ToFloat32()
}

// sliceLastQ reshapes [n, q, hidden] to [n, hidden] by keeping the last query
// position. With q == 1 it is just a reshape; otherwise it gathers position
// q-1 via a reshape-and-take over the flattened (n*q) axis.
func sliceLastQ(h mlxgo.Array, n, q, hidden int) (mlxgo.Array, error) {
	if q == 1 {
		return mlxgo.Reshape(h, []int{n, hidden})
	}
	flat, err := mlxgo.Reshape(h, []int{n * q, hidden})
	if err != nil {
		return mlxgo.Array{}, err
	}
	idx := make([]byte, 4*n)
	for i := range n {
		u := uint32(i*q + q - 1)
		idx[i*4] = byte(u)
		idx[i*4+1] = byte(u >> 8)
		idx[i*4+2] = byte(u >> 16)
		idx[i*4+3] = byte(u >> 24)
	}
	rows, err := mlxgo.FromRawBytes([]int{n}, mlxgo.I32, idx)
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.Take(flat, rows)
}

// prefillMask builds the additive attention mask for the prompt pass, shaped
// [n, heads, P, P]. A position (query i, key j) is allowed when the key is
// causal (j <= i) and not left padding (j >= pad). Padding query rows are given
// a single valid key so their softmax is defined and never produces NaN; their
// outputs are discarded. The mask is replicated across heads rather than left
// as a size-1 axis: the fused attention kernel does not index the batch axis of
// a mask whose head axis is broadcast, which silently applies one sequence's
// mask to the whole batch.
func (b *Batch) prefillMask(P int) (mlxgo.Array, error) {
	h := b.m.Args.NumAttentionHeads
	rows := make([]float32, b.n*P*P)
	for s := range b.n {
		pad := b.pads[s]
		base := s * P * P
		for i := range P {
			row := base + i*P
			for j := range P {
				if (j <= i && j >= pad) || j == i {
					rows[row+j] = 0
				} else {
					rows[row+j] = maskNeg
				}
			}
		}
	}
	return b.maskArray(replicateHeads(rows, b.n, h, P*P), []int{b.n, h, P, P})
}

// decodeMask builds the additive mask for a single decode step, shaped
// [n, heads, 1, L]. The lone query attends to every real key (index >= pad)
// across the full cache length L.
func (b *Batch) decodeMask(L int) (mlxgo.Array, error) {
	h := b.m.Args.NumAttentionHeads
	rows := make([]float32, b.n*L)
	for s := range b.n {
		pad := b.pads[s]
		base := s * L
		for j := range L {
			if j >= pad {
				rows[base+j] = 0
			} else {
				rows[base+j] = maskNeg
			}
		}
	}
	return b.maskArray(replicateHeads(rows, b.n, h, L), []int{b.n, h, 1, L})
}

// replicateHeads expands a per-sequence mask of stride floats into one value
// per head, producing the [n, heads, ...] layout the kernel indexes correctly.
func replicateHeads(rows []float32, n, heads, stride int) []float32 {
	out := make([]float32, n*heads*stride)
	for s := range n {
		src := rows[s*stride : (s+1)*stride]
		for hh := range heads {
			copy(out[(s*heads+hh)*stride:], src)
		}
	}
	return out
}

// maskArray uploads a float32 mask and casts it to the model's compute dtype so
// the fused attention kernel accepts it alongside bf16/f16 inputs.
func (b *Batch) maskArray(data []float32, shape []int) (mlxgo.Array, error) {
	m, err := mlxgo.FromFloat32(shape, data)
	if err != nil {
		return mlxgo.Array{}, err
	}
	dt := b.m.Embed.DType()
	if dt == mlxgo.F32 {
		return m, nil
	}
	return mlxgo.Astype(m, dt)
}

// attention computes grouped-query masked attention over a batch. The fused
// kernel silently drops the additive mask when the query length is below a
// tile threshold, which corrupts the prefill of short prompts, so the batch
// path spells the attention out with plain matmuls. Query heads are folded into
// a (kv_head, group) pair so each key/value head broadcasts over the query heads
// that share it; the mask, replicated per query head upstream, reshapes the same
// way. qL is the query length and kL the key length (cache length).
// initScale builds the attention scale as a one-element array in the model's
// compute dtype, so multiplying the bf16 scores keeps them in bf16.
func (b *Batch) initScale() error {
	s, err := mlxgo.FromFloat32([]int{1}, []float32{b.m.scale})
	if err != nil {
		return err
	}
	if dt := b.m.Embed.DType(); dt != mlxgo.F32 {
		if s, err = mlxgo.Astype(s, dt); err != nil {
			return err
		}
	}
	b.scaleArr = s
	return nil
}

func (b *Batch) attention(qp, kp, vp, mask mlxgo.Array, qL, kL int) (mlxgo.Array, error) {
	a := b.m.Args
	n := b.n
	qh := a.NumAttentionHeads
	kvh := a.NumKeyValueHeads
	rep := qh / kvh
	dim := a.HeadDim

	q5, err := mlxgo.Reshape(qp, []int{n, kvh, rep, qL, dim})
	if err != nil {
		return mlxgo.Array{}, err
	}
	k5, err := mlxgo.Reshape(kp, []int{n, kvh, 1, kL, dim})
	if err != nil {
		return mlxgo.Array{}, err
	}
	v5, err := mlxgo.Reshape(vp, []int{n, kvh, 1, kL, dim})
	if err != nil {
		return mlxgo.Array{}, err
	}
	kT, err := mlxgo.Transpose(k5, []int{0, 1, 2, 4, 3}) // [n, kvh, 1, dim, kL]
	if err != nil {
		return mlxgo.Array{}, err
	}
	scores, err := mlxgo.MatMul(q5, kT) // [n, kvh, rep, qL, kL]
	if err != nil {
		return mlxgo.Array{}, err
	}
	if scores, err = mlxgo.Multiply(scores, b.scaleArr); err != nil {
		return mlxgo.Array{}, err
	}
	m5, err := mlxgo.Reshape(mask, []int{n, kvh, rep, qL, kL})
	if err != nil {
		return mlxgo.Array{}, err
	}
	if scores, err = mlxgo.Add(scores, m5); err != nil {
		return mlxgo.Array{}, err
	}
	if scores, err = mlxgo.SoftmaxAxis(scores, 4); err != nil {
		return mlxgo.Array{}, err
	}
	out, err := mlxgo.MatMul(scores, v5) // [n, kvh, rep, qL, dim]
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.Reshape(out, []int{n, qh, qL, dim})
}

// toHeadsB reshapes [n, q, heads*dim] to [n, heads, q, dim].
func toHeadsB(x mlxgo.Array, n, q, heads, dim int) (mlxgo.Array, error) {
	r, err := mlxgo.Reshape(x, []int{n, q, heads, dim})
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.Transpose(r, []int{0, 2, 1, 3})
}

// fromHeadsB reshapes [n, heads, q, dim] back to [n, q, heads*dim].
func fromHeadsB(x mlxgo.Array, n, q, heads, dim int) (mlxgo.Array, error) {
	t, err := mlxgo.Transpose(x, []int{0, 2, 1, 3})
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.Reshape(t, []int{n, q, heads * dim})
}
