// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"math"
	"testing"
)

func approxSlice(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i])-float64(want[i])) > tol {
			t.Errorf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestLogitsRepetitionPenalty(t *testing.T) {
	logits := []float32{2, -1, 0.5}
	p := LogitsProcessor{RepetitionPenalty: 2}
	p.Apply(logits, []int{0, 1})
	// Positive logit divided, negative multiplied, untouched token unchanged.
	approxSlice(t, logits, []float32{1.0, -2.0, 0.5}, 1e-6)
}

func TestLogitsRepetitionContextWindow(t *testing.T) {
	logits := []float32{1, 1, 1, 1, 1}
	p := LogitsProcessor{RepetitionPenalty: 2, RepetitionContextSize: 2}
	p.Apply(logits, []int{0, 1, 2, 3, 4})
	// Only the last two tokens (3, 4) fall inside the window.
	approxSlice(t, logits, []float32{1, 1, 1, 0.5, 0.5}, 1e-6)
}

func TestLogitsPresenceAndFrequency(t *testing.T) {
	logits := []float32{1, 1, 1, 1}
	p := LogitsProcessor{PresencePenalty: 0.3, FrequencyPenalty: 0.5}
	p.Apply(logits, []int{0, 0, 1})
	// token 0: count 2 -> 1 - (0.5*2 + 0.3); token 1: count 1 -> 1 - (0.5 + 0.3).
	approxSlice(t, logits, []float32{-0.3, 0.2, 1, 1}, 1e-6)
}

func TestLogitsNoOpAtIdentity(t *testing.T) {
	logits := []float32{0.1, 0.2, 0.3}
	orig := append([]float32(nil), logits...)
	(LogitsProcessor{RepetitionPenalty: 1}).Apply(logits, []int{0, 1, 2})
	approxSlice(t, logits, orig, 0)
	if (LogitsProcessor{RepetitionPenalty: 1}).Enabled() {
		t.Error("identity processor reports enabled")
	}
}

func TestLogitsIgnoresOutOfRangeTokens(t *testing.T) {
	logits := []float32{1, 1}
	p := LogitsProcessor{RepetitionPenalty: 2, PresencePenalty: 1}
	// Token ids past the vocab must be skipped, not panic.
	p.Apply(logits, []int{5, -3, 0})
	if logits[1] != 1 {
		t.Errorf("untouched token changed: %v", logits[1])
	}
}
