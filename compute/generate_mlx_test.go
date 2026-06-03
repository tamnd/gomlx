// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package compute

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/gomlx/tokenizer"
)

// modelDir returns the local Qwen3-0.6B directory, or skips the test when it is
// not present. Point GOMLX_QWEN3_DIR at a model directory to run elsewhere.
func modelDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("GOMLX_QWEN3_DIR")
	if dir == "" {
		dir = filepath.Join(os.Getenv("HOME"), "models", "qwen3-0.6b")
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("Qwen3 model not present at %s", dir)
	}
	return dir
}

func loadGenerator(t *testing.T, dir string) *Generator {
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
	return &Generator{Model: model, Tok: tok, EOS: []int{151645, 151643}}
}

// TestQwen3FirstToken loads the real model and greedily decodes a short reply.
// It is the end-to-end proof that the forward pass produces real tokens: the
// output is logged so coherence can be eyeballed, and the test asserts that
// generation produced tokens and stopped cleanly.
func TestQwen3FirstToken(t *testing.T) {
	dir := modelDir(t)
	g := loadGenerator(t, dir)

	prompt := tokenizer.ApplyChatML([]tokenizer.ChatMsg{
		{Role: "user", Content: "Reply with a short friendly greeting."},
	}, true)

	res, err := g.Generate(prompt, GenConfig{
		MaxTokens: 64,
		Sampler:   Sampler{Temperature: 0}, // greedy
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	t.Logf("prompt tokens: %d", res.PromptTokens)
	t.Logf("completion tokens: %d", res.CompletionTokens)
	t.Logf("finish reason: %s", res.FinishReason)
	t.Logf("output: %q", res.Text)

	if res.CompletionTokens == 0 {
		t.Fatal("no tokens generated")
	}
	if res.Text == "" {
		t.Fatal("empty output text")
	}
}

// TestQwen3GreedyDeterministic checks that greedy decoding is reproducible: the
// same prompt yields the same first tokens on two runs.
func TestQwen3GreedyDeterministic(t *testing.T) {
	dir := modelDir(t)
	g := loadGenerator(t, dir)
	prompt := tokenizer.ApplyChatML([]tokenizer.ChatMsg{
		{Role: "user", Content: "Name one color."},
	}, true)

	cfg := GenConfig{MaxTokens: 8, Sampler: Sampler{Temperature: 0}}
	a, err := g.Generate(prompt, cfg)
	if err != nil {
		t.Fatalf("run a: %v", err)
	}
	b, err := g.Generate(prompt, cfg)
	if err != nil {
		t.Fatalf("run b: %v", err)
	}
	if len(a.Tokens) == 0 || len(b.Tokens) == 0 {
		t.Fatal("no tokens")
	}
	n := min(len(a.Tokens), len(b.Tokens))
	for i := 0; i < n; i++ {
		if a.Tokens[i] != b.Tokens[i] {
			t.Fatalf("nondeterministic at %d: %v vs %v", i, a.Tokens[:n], b.Tokens[:n])
		}
	}
}
