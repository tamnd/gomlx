// SPDX-License-Identifier: Apache-2.0

package engine

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/tamnd/gomlx/compute"
	"github.com/tamnd/gomlx/tokenizer"
)

// MLXEngine is the GPU-backed Engine. It loads a Qwen3 checkpoint and runs the
// pure-Go forward pass over MLX. Generation is serialized with a mutex: the
// device runs one request at a time, and the scheduler layers batching on top.
type MLXEngine struct {
	name string
	gen  *compute.Generator
	mu   sync.Mutex
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
	args, err := compute.LoadQwen3Args(cfgRaw)
	if err != nil {
		return nil, fmt.Errorf("mlx engine: parse config: %w", err)
	}

	st, err := compute.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("mlx engine: open weights: %w", err)
	}
	model, err := compute.NewQwen3Model(st, args)
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
	return &MLXEngine{
		name: name,
		gen:  &compute.Generator{Model: model, Tok: tok, EOS: qwen3EOS},
	}, nil
}

// qwen3EOS lists the token ids that end a Qwen3 turn: <|endoftext|> and
// <|im_end|>.
var qwen3EOS = []int{151645, 151643}

func (e *MLXEngine) ModelName() string           { return e.name }
func (e *MLXEngine) IsMLLM() bool                { return false }
func (e *MLXEngine) Start(context.Context) error { return nil }
func (e *MLXEngine) Stop(context.Context) error  { return nil }

// EstimateNewTokens encodes the prompt to count tokens exactly.
func (e *MLXEngine) EstimateNewTokens(prompt string) (int, int) {
	n := len(e.gen.Tok.Encode(prompt))
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
	e.mu.Lock()
	defer e.mu.Unlock()
	res, err := e.gen.Generate(prompt, genConfig(ctx, p, nil))
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
		e.mu.Lock()
		defer e.mu.Unlock()

		emit := func(delta string) {
			select {
			case <-ctx.Done():
			case ch <- GenerationOutput{NewText: delta, Channel: ChannelContent}:
			}
		}
		res, err := e.gen.Generate(prompt, genConfig(ctx, p, emit))
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
