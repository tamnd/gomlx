// SPDX-License-Identifier: Apache-2.0

// Package speculative holds the draft proposers that let the engine guess several
// tokens ahead and verify them in one model step, so a run of easy tokens costs
// far less than one step each. The first proposer here needs no draft model at
// all: prompt lookup finds where the most recent few tokens occurred earlier in
// the context and proposes whatever followed them last time. That pays off
// whenever the output repeats the input, which is common in code, structured
// formats like JSON, and quoting or translation, and it costs nothing when no
// repeat exists because the proposal is simply empty.
package speculative

import "encoding/binary"

// Defaults applied when a constructor argument is non-positive.
const (
	DefaultNumDraft   = 4
	DefaultNgramSize  = 3
	DefaultMinMatches = 2
)

// PromptLookup proposes draft tokens by matching the most recent ngramSize tokens
// against earlier occurrences in the token history and returning what followed.
// It keeps an index from each ngramSize-token window to the positions where that
// window started, so a lookup is a single map probe rather than a scan.
//
// PromptLookup is owned by one sequence and is not safe for concurrent use.
type PromptLookup struct {
	numDraft   int
	ngramSize  int
	minMatches int

	history []int32
	index   map[string][]int // ngram window -> start positions

	stats Stats
}

// Stats reports how the proposer has done so far.
type Stats struct {
	TotalDrafts      int // lookups that returned a non-empty draft
	SuccessfulDrafts int // drafts that had at least one token accepted
	TotalDraftTokens int // tokens proposed across all drafts
	AcceptedTokens   int // proposed tokens the model later confirmed
	HistorySize      int
}

// AcceptanceRate is the share of proposed tokens that were accepted, or zero when
// nothing has been proposed.
func (s Stats) AcceptanceRate() float64 {
	if s.TotalDraftTokens == 0 {
		return 0
	}
	return float64(s.AcceptedTokens) / float64(s.TotalDraftTokens)
}

// NewPromptLookup returns a proposer drafting up to numDraft tokens per step from
// ngramSize-token matches, returning a draft only when it is at least minMatches
// long. Non-positive arguments fall back to the package defaults.
func NewPromptLookup(numDraft, ngramSize, minMatches int) *PromptLookup {
	if numDraft <= 0 {
		numDraft = DefaultNumDraft
	}
	if ngramSize <= 0 {
		ngramSize = DefaultNgramSize
	}
	if minMatches <= 0 {
		minMatches = DefaultMinMatches
	}
	return &PromptLookup{
		numDraft:   numDraft,
		ngramSize:  ngramSize,
		minMatches: minMatches,
		index:      make(map[string][]int),
	}
}

// Reset clears all history and statistics for a fresh generation.
func (p *PromptLookup) Reset() {
	p.history = nil
	p.index = make(map[string][]int)
	p.stats = Stats{}
}

// AddPrompt seeds the history with the prompt tokens.
func (p *PromptLookup) AddPrompt(tokens []int32) {
	for _, t := range tokens {
		p.add(t)
	}
}

// AddToken appends one token, typically one the model just generated, so later
// lookups can match against it.
func (p *PromptLookup) AddToken(t int32) { p.add(t) }

// add appends a token and records the ngramSize-window ending at it.
func (p *PromptLookup) add(t int32) {
	p.history = append(p.history, t)
	p.stats.HistorySize = len(p.history)
	pos := len(p.history) - 1
	start := pos - p.ngramSize + 1
	if start >= 0 {
		k := ngramKey(p.history[start : pos+1])
		p.index[k] = append(p.index[k], start)
	}
}

// Draft returns the proposed tokens for the current history, or nil when there is
// no usable match. It picks the earlier occurrence whose continuation is longest,
// capped at numDraft, and returns nothing if that continuation is shorter than
// minMatches. A returned draft is counted toward the statistics.
func (p *PromptLookup) Draft() []int32 {
	if len(p.history) < p.ngramSize {
		return nil
	}
	currentStart := len(p.history) - p.ngramSize
	query := ngramKey(p.history[currentStart:])
	positions := p.index[query]
	if len(positions) == 0 {
		return nil
	}

	var best []int32
	for _, start := range positions {
		if start == currentStart {
			continue // the query's own occurrence
		}
		begin := start + p.ngramSize
		end := min(begin+p.numDraft, len(p.history))
		if cont := p.history[begin:end]; len(cont) > len(best) {
			best = cont
		}
	}

	if len(best) < p.minMatches {
		return nil
	}
	draft := make([]int32, len(best))
	copy(draft, best)
	p.stats.TotalDrafts++
	p.stats.TotalDraftTokens += len(draft)
	return draft
}

// RecordAccepted updates the statistics with how many of the last draft's tokens
// the model confirmed.
func (p *PromptLookup) RecordAccepted(n int) {
	if n > 0 {
		p.stats.SuccessfulDrafts++
		p.stats.AcceptedTokens += n
	}
}

// Stats returns a copy of the current statistics.
func (p *PromptLookup) Stats() Stats { return p.stats }

// ngramKey encodes a token window into a map key. Little-endian fixed-width
// encoding keeps distinct windows distinct without a hash collision to reason
// about.
func ngramKey(window []int32) string {
	b := make([]byte, len(window)*4)
	for i, t := range window {
		binary.LittleEndian.PutUint32(b[i*4:], uint32(t))
	}
	return string(b)
}
