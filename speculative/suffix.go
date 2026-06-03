// SPDX-License-Identifier: Apache-2.0

package speculative

// Suffix decoding is a stronger draft-model-free proposer than prompt lookup.
// Rather than trusting the first place the recent tokens appeared, it looks at
// every earlier occurrence of the current suffix and lets them vote on what comes
// next, position by position. It drafts the winning token while the vote is
// confident and stops the moment the matches disagree, so a long unambiguous
// repeat produces a long draft and an ambiguous one produces a short, safe draft
// instead of a guess that will be rejected. It prefers the longest matching
// suffix it can find, since a longer match is more discriminating.

// Suffix-decoder defaults applied when a constructor argument is out of range.
const (
	DefaultMaxDraftTokens = 8
	DefaultMaxSuffixLen   = 4
	DefaultMinConfidence  = 0.3
	DefaultMaxHistory     = 32000
)

// DraftStats is the suffix decoder's bookkeeping.
type DraftStats struct {
	TotalDraftsProposed      int
	TotalDraftTokensProposed int
	TotalDraftTokensAccepted int
	SumDraftLength           int
	NStepCalls               int // times a draft was requested
	NDraftsReturned          int // requests that returned a non-empty draft
}

// AcceptanceRate is the share of proposed tokens later accepted.
func (s DraftStats) AcceptanceRate() float64 {
	if s.TotalDraftTokensProposed == 0 {
		return 0
	}
	return float64(s.TotalDraftTokensAccepted) / float64(s.TotalDraftTokensProposed)
}

// MeanAcceptedPerStep is accepted tokens per drafting step. A value above zero is
// roughly the extra decode throughput speculation buys: 0.7 means about 1.7
// tokens emitted per verify step instead of one.
func (s DraftStats) MeanAcceptedPerStep() float64 {
	if s.NStepCalls == 0 {
		return 0
	}
	return float64(s.TotalDraftTokensAccepted) / float64(s.NStepCalls)
}

// SuffixDecoder proposes variable-length drafts from a suffix index over the
// prompt and generated tokens. It is owned by one sequence and is not safe for
// concurrent use.
type SuffixDecoder struct {
	maxDraftTokens int
	maxSuffixLen   int
	minConfidence  float64
	maxHistory     int // <= 0 disables history trimming

	tokens []int32
	shift  int // tokens dropped from the head; index positions are absolute

	// index[k] maps a k-token window to the absolute end positions where it
	// occurs. index[0] is unused so a length k indexes at index[k].
	index []map[string][]int

	stats DraftStats
}

// NewSuffixDecoder returns a decoder capping drafts at maxDraftTokens, indexing
// suffixes up to maxSuffixLen tokens, requiring votes to agree at minConfidence
// (a fraction in [0, 1]) before drafting a position, and trimming history beyond
// maxHistory tokens. A non-positive maxHistory disables trimming. Out-of-range
// arguments fall back to the package defaults.
func NewSuffixDecoder(maxDraftTokens, maxSuffixLen int, minConfidence float64, maxHistory int) *SuffixDecoder {
	if maxDraftTokens < 1 {
		maxDraftTokens = DefaultMaxDraftTokens
	}
	if maxSuffixLen < 1 {
		maxSuffixLen = DefaultMaxSuffixLen
	}
	if minConfidence < 0 || minConfidence > 1 {
		minConfidence = DefaultMinConfidence
	}
	index := make([]map[string][]int, maxSuffixLen+1)
	for k := 1; k <= maxSuffixLen; k++ {
		index[k] = make(map[string][]int)
	}
	return &SuffixDecoder{
		maxDraftTokens: maxDraftTokens,
		maxSuffixLen:   maxSuffixLen,
		minConfidence:  minConfidence,
		maxHistory:     maxHistory,
		index:          index,
	}
}

// Reset clears all history, index, and statistics.
func (d *SuffixDecoder) Reset() {
	d.tokens = nil
	d.shift = 0
	for k := 1; k <= d.maxSuffixLen; k++ {
		d.index[k] = make(map[string][]int)
	}
	d.stats = DraftStats{}
}

// AddPrompt indexes the prompt tokens before generation begins.
func (d *SuffixDecoder) AddPrompt(tokens []int32) {
	for _, t := range tokens {
		d.add(t)
	}
}

// AddToken indexes one newly generated token.
func (d *SuffixDecoder) AddToken(t int32) { d.add(t) }

func (d *SuffixDecoder) add(t int32) {
	d.tokens = append(d.tokens, t)
	absPos := d.shift + len(d.tokens) - 1
	for k := 1; k <= d.maxSuffixLen; k++ {
		start := len(d.tokens) - k
		if start < 0 {
			continue
		}
		key := ngramKey(d.tokens[start : start+k])
		d.index[k][key] = append(d.index[k][key], absPos)
	}
	d.trim()
}

// trim drops the oldest tokens once history exceeds the cap and removes the index
// entries that point only into the dropped region, so a stale position can never
// resolve to the wrong token in the shifted window.
func (d *SuffixDecoder) trim() {
	if d.maxHistory <= 0 || len(d.tokens) <= d.maxHistory {
		return
	}
	drop := len(d.tokens) - d.maxHistory
	d.tokens = append([]int32(nil), d.tokens[drop:]...)
	d.shift += drop
	for k := 1; k <= d.maxSuffixLen; k++ {
		bucket := d.index[k]
		// A k-gram is still valid only when its whole span survives, i.e. its
		// end position is at least shift + k - 1.
		threshold := d.shift + k - 1
		for key, ends := range bucket {
			fresh := ends[:0]
			for _, e := range ends {
				if e >= threshold {
					fresh = append(fresh, e)
				}
			}
			if len(fresh) == 0 {
				delete(bucket, key)
			} else {
				bucket[key] = fresh
			}
		}
	}
}

// Draft proposes a continuation for the current history, preferring the longest
// matching suffix. It returns nil when no suffix has an earlier match that clears
// the confidence floor.
func (d *SuffixDecoder) Draft() []int32 {
	d.stats.NStepCalls++
	if len(d.tokens) == 0 {
		return nil
	}
	maxK := min(d.maxSuffixLen, len(d.tokens))
	for k := maxK; k >= 1; k-- {
		query := ngramKey(d.tokens[len(d.tokens)-k:])
		positions := d.index[k][query]
		if len(positions) == 0 {
			continue
		}
		currentEnd := d.shift + len(d.tokens) - 1
		if draft := d.buildDraft(positions, currentEnd); len(draft) > 0 {
			d.stats.TotalDraftsProposed++
			d.stats.TotalDraftTokensProposed += len(draft)
			d.stats.NDraftsReturned++
			d.stats.SumDraftLength += len(draft)
			return draft
		}
	}
	return nil
}

// buildDraft votes over the continuations following each match position and
// appends the winning token while the vote clears minConfidence, narrowing the
// match set to the agreeing positions each step so a drifting match drops out
// quickly.
func (d *SuffixDecoder) buildDraft(positions []int, currentEnd int) []int32 {
	var draft []int32
	matches := positions
	for offset := 0; offset < d.maxDraftTokens; offset++ {
		counts := map[int32]int{}
		total := 0
		for _, end := range matches {
			if end == currentEnd {
				continue
			}
			local := end + 1 + offset - d.shift
			if local >= 0 && local < len(d.tokens) {
				counts[d.tokens[local]]++
				total++
			}
		}
		if total == 0 {
			break
		}
		top, topCount := topToken(counts)
		if float64(topCount)/float64(total) < d.minConfidence {
			break
		}
		draft = append(draft, top)

		// Keep only the positions that produced the winning token. This must be
		// a fresh slice: on the first pass matches aliases the slice stored in
		// the index, and filtering in place would corrupt it.
		kept := make([]int, 0, len(matches))
		for _, end := range matches {
			if end == currentEnd {
				continue
			}
			local := end + 1 + offset - d.shift
			if local >= 0 && local < len(d.tokens) && d.tokens[local] == top {
				kept = append(kept, end)
			}
		}
		matches = kept
		if len(matches) == 0 {
			break
		}
	}
	return draft
}

// RecordAccepted records how many of the last draft's tokens the model confirmed.
func (d *SuffixDecoder) RecordAccepted(n int) {
	if n > 0 {
		d.stats.TotalDraftTokensAccepted += n
	}
}

// Stats returns a copy of the decoder's statistics.
func (d *SuffixDecoder) Stats() DraftStats { return d.stats }

// topToken returns the most-voted token, breaking ties toward the smaller token
// id so the result is deterministic regardless of map iteration order.
func topToken(counts map[int32]int) (top int32, count int) {
	first := true
	for tok, c := range counts {
		if first || c > count || (c == count && tok < top) {
			top, count = tok, c
			first = false
		}
	}
	return top, count
}
