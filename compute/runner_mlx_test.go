// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package compute

import (
	"sync"
	"testing"
)

// TestRunnerConcurrentMatchesSingle fires many requests at the Runner at once
// and checks every response is routed back to the right caller with the right
// content. Greedy decoding lets us compare against the single-stream Generator,
// but only on a token prefix: the batched path uses a plain-matmul attention
// while the single-stream path uses the fused kernel, and the two round bf16
// differently, so once the model hits a near-tie the two equally likely
// continuations may diverge. The prefix is well clear of any tie for these
// prompts, so an exact prefix match proves the queue plumbing preserves
// per-request output regardless of how requests are grouped into batches.
func TestRunnerConcurrentMatchesSingle(t *testing.T) {
	dir := modelDir(t)
	m, tok := loadModel(t, dir)

	gen := &Generator{Model: m, Tok: tok, EOS: []int{151645, 151643}}
	prompts := []string{
		"The capital of France is",
		"Water is made of hydrogen and",
		"The opposite of hot is",
		"Two plus two equals",
	}
	cfg := func() GenConfig { return GenConfig{MaxTokens: 16} }

	const prefix = 8
	want := make([][]int, len(prompts))
	for i, p := range prompts {
		res, err := gen.Generate(p, cfg())
		if err != nil {
			t.Fatalf("single generate: %v", err)
		}
		want[i] = res.Tokens[:prefix]
	}

	runner := &Runner{Model: m, Tok: tok, EOS: []int{151645, 151643}, MaxBatch: 4}
	runner.Start()

	const repeats = 3
	var wg sync.WaitGroup
	errs := make(chan error, len(prompts)*repeats)
	for r := 0; r < repeats; r++ {
		for i, p := range prompts {
			wg.Add(1)
			go func(i int, p string) {
				defer wg.Done()
				res, err := runner.Generate(p, cfg())
				if err != nil {
					errs <- err
					return
				}
				if len(res.Tokens) < prefix || !equalInts(res.Tokens[:prefix], want[i]) {
					t.Errorf("prompt %q\n batch=%v\nsingle=%v", p, res.Tokens, want[i])
				}
			}(i, p)
		}
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("runner generate: %v", err)
	}
}

// TestRunnerUnpaddedMatchesSingle batches several copies of one prompt, so every
// sequence shares a length and the batch carries no left padding. The decoded
// prefix must match the single-stream Generator, covering the equal-length batch
// shape alongside the mixed-length case in TestRunnerConcurrentMatchesSingle.
func TestRunnerUnpaddedMatchesSingle(t *testing.T) {
	dir := modelDir(t)
	m, tok := loadModel(t, dir)

	const prompt = "The capital of France is"
	const prefix = 8
	cfg := func() GenConfig { return GenConfig{MaxTokens: 16} }

	gen := &Generator{Model: m, Tok: tok, EOS: []int{151645, 151643}}
	single, err := gen.Generate(prompt, cfg())
	if err != nil {
		t.Fatalf("single generate: %v", err)
	}
	want := single.Tokens[:prefix]

	runner := &Runner{Model: m, Tok: tok, EOS: []int{151645, 151643}, MaxBatch: 6}
	runner.Start()

	const n = 6
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for k := 0; k < n; k++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := runner.Generate(prompt, cfg())
			if err != nil {
				errs <- err
				return
			}
			if len(res.Tokens) < prefix || !equalInts(res.Tokens[:prefix], want) {
				t.Errorf("batch=%v\nsingle=%v", res.Tokens, want)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("runner generate: %v", err)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
