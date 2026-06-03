// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package engine

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func mlxModelDir(t *testing.T) string {
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

// TestMLXEngineChat exercises the Engine adapter on the GPU: the constructor
// loads the checkpoint, Chat renders a ChatML prompt and returns a reply, and
// StreamChat streams deltas that reassemble into the same kind of output.
func TestMLXEngineChat(t *testing.T) {
	dir := mlxModelDir(t)
	eng, err := NewMLXEngine("qwen3-0.6b", dir)
	if err != nil {
		t.Fatalf("NewMLXEngine: %v", err)
	}
	if eng.ModelName() != "qwen3-0.6b" {
		t.Errorf("model name: got %q", eng.ModelName())
	}

	msgs := []ChatMessage{{Role: "user", Content: "Say hello in one word."}}
	p := DefaultSamplingParams()
	p.MaxTokens = 32
	p.Temperature = 0

	out, err := eng.Chat(context.Background(), msgs, p, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if out.CompletionTokens == 0 || out.Text == "" {
		t.Fatalf("empty chat output: %+v", out)
	}
	t.Logf("chat output: %q (%d tokens, %s)", out.Text, out.CompletionTokens, out.FinishReason)

	ch, err := eng.StreamChat(context.Background(), msgs, p, nil)
	if err != nil {
		t.Fatalf("StreamChat: %v", err)
	}
	var sb strings.Builder
	var final GenerationOutput
	for o := range ch {
		if o.Finished {
			final = o
			continue
		}
		sb.WriteString(o.NewText)
	}
	if final.CompletionTokens == 0 {
		t.Fatalf("stream produced no final marker")
	}
	if sb.Len() == 0 {
		t.Fatalf("stream produced no text deltas")
	}
	t.Logf("stream output: %q (%d tokens)", sb.String(), final.CompletionTokens)
}

// TestMLXEngineCancel checks that a cancelled context stops streaming.
func TestMLXEngineCancel(t *testing.T) {
	dir := mlxModelDir(t)
	eng, err := NewMLXEngine("qwen3-0.6b", dir)
	if err != nil {
		t.Fatalf("NewMLXEngine: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := DefaultSamplingParams()
	p.MaxTokens = 256
	p.Temperature = 0

	ch, err := eng.StreamGenerate(ctx, "Count slowly: one, two,", p)
	if err != nil {
		t.Fatalf("StreamGenerate: %v", err)
	}
	got := 0
	for range ch {
		got++
		if got == 2 {
			cancel()
		}
	}
	// The channel must close after cancellation rather than running to the full
	// token budget; a couple extra tokens may slip through before the check.
	if got > 20 {
		t.Errorf("cancellation did not stop generation promptly: %d events", got)
	}
}
