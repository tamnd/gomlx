// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"math/rand"
	"strings"

	"github.com/tamnd/gomlx/mlxgo"
	"github.com/tamnd/gomlx/tokenizer"
)

// Generator runs autoregressive decoding for a loaded Qwen3 model. It owns the
// model and tokenizer and is reused across requests; per-request state (the KV
// cache and sampling RNG) is created inside Generate.
type Generator struct {
	Model *Qwen3Model
	Tok   *tokenizer.Tokenizer
	EOS   []int
}

// GenConfig controls one generation run.
type GenConfig struct {
	MaxTokens int
	Sampler   Sampler
	Logits    LogitsProcessor
	Stop      []string
	Seed      int64
}

// GenResult is the outcome of a generation run.
type GenResult struct {
	Text             string
	Tokens           []int
	PromptTokens     int
	CompletionTokens int
	FinishReason     string
}

// Generate encodes the prompt, runs the prompt pass, then decodes one token at
// a time until an end token, a stop string, or the token budget is hit. The KV
// cache makes each decode step process only the new token.
func (g *Generator) Generate(prompt string, cfg GenConfig) (GenResult, error) {
	promptIDs := g.Tok.Encode(prompt)
	res := GenResult{PromptTokens: len(promptIDs)}

	caches := g.Model.NewCaches()
	offset := 0

	cur := toInt32(promptIDs)
	generated := make([]int, 0, cfg.MaxTokens)
	rng := rand.New(rand.NewSource(cfg.Seed))

	eos := make(map[int]bool, len(g.EOS))
	for _, e := range g.EOS {
		eos[e] = true
	}

	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 256
	}

	for step := 0; step < maxTokens; step++ {
		logits, err := g.Model.Forward(cur, caches, offset)
		if err != nil {
			return res, err
		}
		offset += len(cur)

		row, err := lastRow(logits, len(cur))
		if err != nil {
			return res, err
		}

		if cfg.Logits.Enabled() {
			cfg.Logits.Apply(row, append(promptIDs, generated...))
		}
		next := cfg.Sampler.Sample(row, rng)

		if eos[next] {
			res.FinishReason = "stop"
			break
		}
		generated = append(generated, next)

		if stop, cut := hitStopString(g.Tok.Decode(generated), cfg.Stop); stop {
			res.Text = cut
			res.Tokens = generated
			res.CompletionTokens = len(generated)
			res.FinishReason = "stop"
			return res, nil
		}

		cur = []int32{int32(next)}
	}

	if res.FinishReason == "" {
		res.FinishReason = "length"
	}
	res.Tokens = generated
	res.CompletionTokens = len(generated)
	res.Text = g.Tok.Decode(generated)
	return res, nil
}

// lastRow gathers the logits for the final position of a [seq, vocab] array and
// copies them to the host.
func lastRow(logits mlxgo.Array, seq int) ([]float32, error) {
	idx, err := mlxgo.FromRawBytes([]int{1}, mlxgo.I32, int32LE(int32(seq-1)))
	if err != nil {
		return nil, err
	}
	row, err := mlxgo.Take(logits, idx)
	if err != nil {
		return nil, err
	}
	return row.ToFloat32()
}

// hitStopString reports whether text contains any stop sequence and returns the
// text truncated at the earliest one.
func hitStopString(text string, stops []string) (bool, string) {
	best := -1
	for _, s := range stops {
		if s == "" {
			continue
		}
		if i := strings.Index(text, s); i >= 0 && (best < 0 || i < best) {
			best = i
		}
	}
	if best < 0 {
		return false, text
	}
	return true, text[:best]
}

func toInt32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}

func int32LE(v int32) []byte {
	u := uint32(v)
	return []byte{byte(u), byte(u >> 8), byte(u >> 16), byte(u >> 24)}
}
