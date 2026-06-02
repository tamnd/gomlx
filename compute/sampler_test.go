// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"math"
	"math/rand"
	"testing"
)

func TestSamplerGreedy(t *testing.T) {
	logits := []float32{0.1, 3.2, 0.5, 3.1, 2.9}
	s := Sampler{Temperature: 0}
	if got := s.Sample(logits, nil); got != 1 {
		t.Errorf("greedy: got %d want 1", got)
	}
	// Temperature 0 must ignore the rng entirely.
	if got := s.Sample(logits, rand.New(rand.NewSource(99))); got != 1 {
		t.Errorf("greedy with rng: got %d want 1", got)
	}
}

func TestSamplerGreedyTieLowestIndex(t *testing.T) {
	logits := []float32{1, 5, 5, 2}
	if got := (Sampler{}).Sample(logits, nil); got != 1 {
		t.Errorf("tie: got %d want 1 (lowest index)", got)
	}
}

func TestSoftmaxSumsToOne(t *testing.T) {
	probs := Softmax([]float32{1, 2, 3, 4})
	var sum float64
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("softmax sum: got %v want 1", sum)
	}
	// Monotonic: larger logit, larger probability.
	for i := 1; i < len(probs); i++ {
		if probs[i] <= probs[i-1] {
			t.Errorf("softmax not monotonic at %d: %v", i, probs)
		}
	}
}

func TestSamplerTopKRestrictsSupport(t *testing.T) {
	// Index 0 dominates; with TopK=1 only it can ever be drawn.
	logits := []float32{10, 1, 1, 1}
	s := Sampler{Temperature: 1, TopK: 1}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 200; i++ {
		if got := s.Sample(logits, rng); got != 0 {
			t.Fatalf("top-k=1 drew %d, expected only 0", got)
		}
	}
}

func TestSamplerTopPRestrictsSupport(t *testing.T) {
	// Softmax mass is concentrated on indices 0 and 1; a modest top-p must
	// never reach the long tail.
	logits := []float32{6, 5, -5, -5, -5}
	s := Sampler{Temperature: 1, TopP: 0.9}
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 500; i++ {
		got := s.Sample(logits, rng)
		if got != 0 && got != 1 {
			t.Fatalf("top-p drew tail token %d", got)
		}
	}
}

func TestSamplerMinPRestrictsSupport(t *testing.T) {
	// min-p keeps only tokens within a fraction of the top probability.
	logits := []float32{6, 5.8, -5, -5}
	s := Sampler{Temperature: 1, MinP: 0.3}
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 500; i++ {
		got := s.Sample(logits, rng)
		if got != 0 && got != 1 {
			t.Fatalf("min-p drew low-probability token %d", got)
		}
	}
}

func TestSamplerDistributionApproxMatchesSoftmax(t *testing.T) {
	// With plain temperature sampling the empirical frequencies should track
	// the softmax probabilities. This guards the categorical draw.
	logits := []float32{2, 1, 0}
	probs := Softmax(logits)
	s := Sampler{Temperature: 1}
	rng := rand.New(rand.NewSource(20260603))
	const n = 60000
	counts := make([]int, len(logits))
	for i := 0; i < n; i++ {
		counts[s.Sample(logits, rng)]++
	}
	for i := range probs {
		freq := float64(counts[i]) / n
		if math.Abs(freq-probs[i]) > 0.02 {
			t.Errorf("token %d freq %.3f, want ~%.3f", i, freq, probs[i])
		}
	}
}

func TestSamplerNilRNGFallsBackToArgmaxOfSupport(t *testing.T) {
	// A non-zero temperature with a nil rng cannot draw randomly, so it should
	// return the most probable surviving candidate.
	logits := []float32{1, 4, 2}
	if got := (Sampler{Temperature: 0.7}).Sample(logits, nil); got != 1 {
		t.Errorf("nil rng: got %d want 1", got)
	}
}
