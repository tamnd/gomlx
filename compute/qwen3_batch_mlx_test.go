// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package compute

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/gomlx/tokenizer"
)

func loadModel(t *testing.T, dir string) (*Qwen3Model, *tokenizer.Tokenizer) {
	t.Helper()
	cfgRaw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	args, err := LoadQwen3Args(cfgRaw)
	if err != nil {
		t.Fatalf("LoadQwen3Args: %v", err)
	}
	st, err := Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatalf("open weights: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	model, err := NewQwen3Model(st, args)
	if err != nil {
		t.Fatalf("NewQwen3Model: %v", err)
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	return model, tok
}

func argmaxRow(logits []float32, row, vocab int) int32 {
	best, bestIdx := logits[row*vocab], 0
	for i := 1; i < vocab; i++ {
		if v := logits[row*vocab+i]; v > best {
			best, bestIdx = v, i
		}
	}
	return int32(bestIdx)
}

// greedySingle decodes steps tokens one sequence at a time through the
// single-stream forward pass, the reference the batched path must match.
func greedySingle(t *testing.T, m *Qwen3Model, prompt []int32, steps, vocab int) []int32 {
	t.Helper()
	caches := m.NewCaches()
	out := make([]int32, 0, steps)
	cur := prompt
	offset := 0
	for s := 0; s < steps; s++ {
		logits, err := m.Forward(cur, caches, offset)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		host, err := logits.ToFloat32()
		if err != nil {
			t.Fatalf("tofloat: %v", err)
		}
		tok := argmaxRow(host, len(cur)-1, vocab)
		out = append(out, tok)
		offset += len(cur)
		cur = []int32{tok}
	}
	return out
}

// greedyBatch decodes steps tokens for all prompts together through the batched
// path, feeding back the per-sequence argmax each step.
func greedyBatch(t *testing.T, m *Qwen3Model, prompts [][]int32, steps, vocab int) [][]int32 {
	t.Helper()
	b := m.NewBatch(len(prompts))
	logits, err := b.Prefill(prompts)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	out := make([][]int32, len(prompts))
	for s := 0; s < steps; s++ {
		next := make([]int32, len(prompts))
		for r := range prompts {
			next[r] = argmaxRow(logits, r, vocab)
			out[r] = append(out[r], next[r])
		}
		if s == steps-1 {
			break
		}
		if logits, err = b.Decode(next); err != nil {
			t.Fatalf("decode step %d: %v", s, err)
		}
	}
	return out
}

// TestBatchMatchesSingleStream checks that batched decode produces exactly the
// same greedy tokens as running each sequence alone, across prompt orderings
// and batch sizes. This exercises the left-padding, the shared rotary offset,
// and the per-sequence attention mask.
func TestBatchMatchesSingleStream(t *testing.T) {
	dir := modelDir(t)
	m, tok := loadModel(t, dir)
	vocab := m.Args.VocabSize

	toI32 := func(s string) []int32 {
		ids := tok.Encode(s)
		out := make([]int32, len(ids))
		for i, v := range ids {
			out[i] = int32(v)
		}
		return out
	}

	short := toI32("The capital of France is")
	long := toI32("Once upon a time, in a small village near the mountains,")
	mid := toI32("The largest planet in the solar system is")
	const steps = 12

	cases := [][][]int32{
		{short, long},      // padded sequence in row 0
		{long, short},      // padded sequence in row 1
		{long, mid, short}, // three sequences, all different lengths
	}

	for ci, prompts := range cases {
		want := make([][]int32, len(prompts))
		for r, p := range prompts {
			want[r] = greedySingle(t, m, p, steps, vocab)
		}
		got := greedyBatch(t, m, prompts, steps, vocab)
		for r := range prompts {
			for i := range want[r] {
				if got[r][i] != want[r][i] {
					t.Errorf("case %d seq %d token %d: batch=%d single=%d",
						ci, r, i, got[r][i], want[r][i])
					break
				}
			}
		}
	}
}
