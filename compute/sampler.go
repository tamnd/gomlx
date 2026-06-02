// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"math"
	"math/rand"
	"sort"
)

// Sampler turns a logits row into a token id. The algorithm mirrors the
// reference make_sampler: temperature 0 is greedy argmax; otherwise the logits
// are temperature-scaled, the top-k, top-p, and min-p filters are applied in
// that order, and a token is drawn from the surviving distribution. The numeric
// work is pure Go so it runs and is tested without a GPU; the on-device path
// reuses the same parameter semantics.
type Sampler struct {
	Temperature float64
	TopK        int     // 0 disables
	TopP        float64 // 0 or >=1 disables
	MinP        float64 // 0 disables
}

// Sample returns the chosen token id for one logits row. rng may be nil when
// the sampler is greedy (Temperature == 0).
func (s Sampler) Sample(logits []float32, rng *rand.Rand) int {
	if len(logits) == 0 {
		return 0
	}
	if s.Temperature <= 0 {
		return argmax(logits)
	}

	// Temperature-scaled softmax over float64 for numerical headroom.
	probs := softmaxScaled(logits, s.Temperature)

	// Candidate index set, full vocabulary to begin with.
	idx := make([]int, len(probs))
	for i := range idx {
		idx[i] = i
	}

	if s.TopK > 0 && s.TopK < len(idx) {
		idx = topKIndices(probs, idx, s.TopK)
	}
	if s.TopP > 0 && s.TopP < 1 {
		idx = topPIndices(probs, idx, s.TopP)
	}
	if s.MinP > 0 {
		idx = minPIndices(probs, idx, s.MinP)
	}

	return sampleFrom(probs, idx, rng)
}

// argmax returns the index of the largest logit, ties broken by lowest index.
func argmax(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

// softmaxScaled returns softmax(logits / temperature) as float64 probabilities.
func softmaxScaled(logits []float32, temp float64) []float64 {
	out := make([]float64, len(logits))
	maxv := math.Inf(-1)
	for _, v := range logits {
		fv := float64(v) / temp
		if fv > maxv {
			maxv = fv
		}
	}
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v)/temp - maxv)
		out[i] = e
		sum += e
	}
	if sum == 0 {
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Softmax returns the plain softmax of logits (temperature 1) as float64.
func Softmax(logits []float32) []float64 { return softmaxScaled(logits, 1) }

// topKIndices keeps the k indices with the highest probability.
func topKIndices(probs []float64, idx []int, k int) []int {
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
	if k > len(idx) {
		k = len(idx)
	}
	return idx[:k]
}

// topPIndices keeps the smallest set of highest-probability indices whose
// cumulative probability reaches p (nucleus sampling).
func topPIndices(probs []float64, idx []int, p float64) []int {
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
	var cum float64
	for i, id := range idx {
		cum += probs[id]
		if cum >= p {
			return idx[:i+1]
		}
	}
	return idx
}

// minPIndices keeps indices whose probability is at least p times the maximum
// probability in the candidate set.
func minPIndices(probs []float64, idx []int, p float64) []int {
	var maxp float64
	for _, id := range idx {
		if probs[id] > maxp {
			maxp = probs[id]
		}
	}
	threshold := p * maxp
	out := idx[:0:0]
	for _, id := range idx {
		if probs[id] >= threshold {
			out = append(out, id)
		}
	}
	if len(out) == 0 {
		return idx
	}
	return out
}

// sampleFrom draws an index from the candidate set in proportion to its
// probability. With a nil rng it returns the highest-probability candidate.
func sampleFrom(probs []float64, idx []int, rng *rand.Rand) int {
	if len(idx) == 0 {
		return 0
	}
	var total float64
	for _, id := range idx {
		total += probs[id]
	}
	if total <= 0 || rng == nil {
		best := idx[0]
		for _, id := range idx {
			if probs[id] > probs[best] {
				best = id
			}
		}
		return best
	}
	r := rng.Float64() * total
	var cum float64
	for _, id := range idx {
		cum += probs[id]
		if r < cum {
			return id
		}
	}
	return idx[len(idx)-1]
}
