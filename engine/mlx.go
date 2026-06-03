// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/tamnd/gomlx/compute"
	"github.com/tamnd/gomlx/tokenizer"
)

// MLXEngine is the GPU-backed Engine. It loads a dense checkpoint (Qwen3, Llama,
// or Mistral) and runs the pure-Go forward pass over MLX. Concurrent requests are
// not serialized: they are handed to a Runner that batches them into a single
// forward pass per step, so throughput under load scales with the batch instead
// of the queue depth.
type MLXEngine struct {
	name string
	tok  *tokenizer.Tokenizer
	run  *compute.Runner
}

// NewMLXEngine loads the model, tokenizer, and config from a directory laid out
// like a Hugging Face snapshot (config.json, tokenizer.json, model.safetensors).
// It returns a clear error when the binary was built without the mlx tag, since
// the weight upload needs a live MLX runtime.
func NewMLXEngine(name, dir string) (*MLXEngine, error) {
	cfgRaw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("mlx engine: read config: %w", err)
	}
	args, err := compute.LoadArgs(cfgRaw)
	if err != nil {
		return nil, fmt.Errorf("mlx engine: parse config: %w", err)
	}

	st, err := compute.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("mlx engine: open weights: %w", err)
	}
	model, err := compute.NewModel(st, args)
	if err != nil {
		st.Close()
		return nil, fmt.Errorf("mlx engine: load weights: %w", err)
	}

	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		st.Close()
		return nil, fmt.Errorf("mlx engine: load tokenizer: %w", err)
	}

	if name == "" {
		name = filepath.Base(dir)
	}
	runner := &compute.Runner{
		Model:    model,
		Tok:      tok,
		EOS:      eosTokens(args),
		MaxBatch: maxBatch(),
	}
	runner.Start()
	return &MLXEngine{name: name, tok: tok, run: runner}, nil
}

// maxBatch reads the decode batch limit from GOMLX_MAX_BATCH, defaulting to 8.
// It caps how many concurrent requests share one forward pass.
func maxBatch() int {
	if v := os.Getenv("GOMLX_MAX_BATCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 8
}

// qwen3EOS lists the token ids that end a Qwen3 turn: <|im_end|> and
// <|endoftext|>. It is the fallback when a config.json omits eos_token_id, which
// shipped Qwen3 checkpoints do not.
var qwen3EOS = []int{151645, 151643}

// eosTokens returns the end-of-turn token ids for a model. It prefers the
// eos_token_id from the checkpoint config, which is correct per family (Llama and
// Mistral use different ids than Qwen3), and falls back to the Qwen3 ids only
// when the config does not declare any.
func eosTokens(args compute.DenseArgs) []int {
	if len(args.EOSTokenIDs) > 0 {
		return args.EOSTokenIDs
	}
	return qwen3EOS
}

func (e *MLXEngine) ModelName() string           { return e.name }
func (e *MLXEngine) IsMLLM() bool                { return false }
func (e *MLXEngine) Start(context.Context) error { return nil }
func (e *MLXEngine) Stop(context.Context) error  { return nil }

// EstimateNewTokens encodes the prompt to count tokens exactly.
func (e *MLXEngine) EstimateNewTokens(prompt string) (int, int) {
	n := len(e.tok.Encode(prompt))
	return n, n
}

// genConfig translates serving sampling params into a compute GenConfig.
func genConfig(ctx context.Context, p SamplingParams, onToken func(string)) compute.GenConfig {
	return compute.GenConfig{
		MaxTokens: p.MaxTokens,
		Sampler: compute.Sampler{
			Temperature: p.Temperature,
			TopK:        p.TopK,
			TopP:        p.TopP,
			MinP:        p.MinP,
		},
		Logits: compute.LogitsProcessor{
			RepetitionPenalty: p.RepetitionPenalty,
			PresencePenalty:   p.PresencePenalty,
			FrequencyPenalty:  p.FrequencyPenalty,
		},
		Stop:    p.Stop,
		Ctx:     ctx,
		OnToken: onToken,
	}
}

func (e *MLXEngine) Generate(ctx context.Context, prompt string, p SamplingParams) (GenerationOutput, error) {
	res, err := e.run.Generate(prompt, genConfig(ctx, p, nil))
	if err != nil {
		return GenerationOutput{}, err
	}
	return GenerationOutput{
		Text:             res.Text,
		Tokens:           res.Tokens,
		PromptTokens:     res.PromptTokens,
		CompletionTokens: res.CompletionTokens,
		FinishReason:     res.FinishReason,
		Finished:         true,
		Channel:          ChannelContent,
	}, nil
}

func (e *MLXEngine) StreamGenerate(ctx context.Context, prompt string, p SamplingParams) (<-chan GenerationOutput, error) {
	ch := make(chan GenerationOutput)
	go func() {
		defer close(ch)

		emit := func(delta string) {
			select {
			case <-ctx.Done():
			case ch <- GenerationOutput{NewText: delta, Channel: ChannelContent}:
			}
		}
		res, err := e.run.Generate(prompt, genConfig(ctx, p, emit))
		if err != nil {
			return
		}
		select {
		case <-ctx.Done():
		case ch <- GenerationOutput{
			Finished:         true,
			FinishReason:     res.FinishReason,
			PromptTokens:     res.PromptTokens,
			CompletionTokens: res.CompletionTokens,
			Channel:          ChannelContent,
		}:
		}
	}()
	return ch, nil
}

func (e *MLXEngine) Chat(ctx context.Context, msgs []ChatMessage, p SamplingParams, _ []any) (GenerationOutput, error) {
	return e.Generate(ctx, e.renderChat(msgs), p)
}

func (e *MLXEngine) StreamChat(ctx context.Context, msgs []ChatMessage, p SamplingParams, _ []any) (<-chan GenerationOutput, error) {
	return e.StreamGenerate(ctx, e.renderChat(msgs), p)
}

// renderChat applies the Qwen3 ChatML template, ending with an open assistant
// turn so the model continues the reply.
func (e *MLXEngine) renderChat(msgs []ChatMessage) string {
	out := make([]tokenizer.ChatMsg, len(msgs))
	for i, m := range msgs {
		out[i] = tokenizer.ChatMsg{Role: m.Role, Content: m.Content}
	}
	return tokenizer.ApplyChatML(out, true)
}

// compile-time check.
var _ Engine = (*MLXEngine)(nil)
