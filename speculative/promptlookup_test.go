// SPDX-License-Identifier: Apache-2.0

package speculative

import (
	"slices"
	"testing"
)

func TestDraftFromRepeatedNgram(t *testing.T) {
	p := NewPromptLookup(4, 3, 1)
	// "1 2 3 9 9 1 2 3": the window [1 2 3] recurs, and last time it was
	// followed by 9 9. After feeding the second [1 2 3], the draft should
	// propose what followed the first occurrence.
	p.AddPrompt([]int32{1, 2, 3, 9, 9, 1, 2, 3})
	draft := p.Draft()
	want := []int32{9, 9, 1, 2} // continuation after the first [1 2 3], capped at 4
	if !slices.Equal(draft, want) {
		t.Fatalf("draft=%v want %v", draft, want)
	}
}

func TestNoMatchYieldsNoDraft(t *testing.T) {
	p := NewPromptLookup(4, 3, 2)
	p.AddPrompt([]int32{1, 2, 3, 4, 5})
	if d := p.Draft(); d != nil {
		t.Fatalf("unique tokens should yield no draft, got %v", d)
	}
}

func TestShortHistoryYieldsNoDraft(t *testing.T) {
	p := NewPromptLookup(4, 3, 1)
	p.AddPrompt([]int32{1, 2}) // shorter than the n-gram size
	if d := p.Draft(); d != nil {
		t.Fatalf("history shorter than ngram should yield no draft, got %v", d)
	}
}

func TestDraftCappedByNumDraft(t *testing.T) {
	p := NewPromptLookup(2, 3, 1) // at most two draft tokens
	p.AddPrompt([]int32{1, 2, 3, 7, 8, 9, 0, 1, 2, 3})
	draft := p.Draft()
	if len(draft) != 2 {
		t.Fatalf("draft length=%d want 2 (capped)", len(draft))
	}
	if !slices.Equal(draft, []int32{7, 8}) {
		t.Fatalf("draft=%v want [7 8]", draft)
	}
}

func TestMinMatchesSuppressesShortContinuation(t *testing.T) {
	// History [1 1 1] with a 2-gram: the window [1 1] recurs, but its earlier
	// occurrence (start 0) is followed by only one token before the history
	// ends, so the continuation is length 1. With minMatches of 2 that is too
	// short to propose.
	p := NewPromptLookup(4, 2, 2)
	p.AddPrompt([]int32{1, 1, 1})
	if d := p.Draft(); d != nil {
		t.Fatalf("continuation shorter than minMatches must not draft, got %v", d)
	}
	// Lowering minMatches to 1 should then accept that single-token draft.
	q := NewPromptLookup(4, 2, 1)
	q.AddPrompt([]int32{1, 1, 1})
	if d := q.Draft(); !slices.Equal(d, []int32{1}) {
		t.Fatalf("draft=%v want [1]", d)
	}
}

func TestStatsTrackAcceptance(t *testing.T) {
	p := NewPromptLookup(4, 3, 1)
	p.AddPrompt([]int32{1, 2, 3, 9, 9, 1, 2, 3})
	draft := p.Draft()
	if len(draft) == 0 {
		t.Fatal("expected a draft to test stats")
	}
	p.RecordAccepted(2) // pretend the model confirmed two of them

	s := p.Stats()
	if s.TotalDrafts != 1 {
		t.Fatalf("total drafts=%d want 1", s.TotalDrafts)
	}
	if s.TotalDraftTokens != len(draft) {
		t.Fatalf("total draft tokens=%d want %d", s.TotalDraftTokens, len(draft))
	}
	if s.SuccessfulDrafts != 1 || s.AcceptedTokens != 2 {
		t.Fatalf("acceptance not tracked: %+v", s)
	}
	if got := s.AcceptanceRate(); got != 2.0/float64(len(draft)) {
		t.Fatalf("rate=%v want %v", got, 2.0/float64(len(draft)))
	}
}

func TestResetClearsState(t *testing.T) {
	p := NewPromptLookup(4, 3, 1)
	p.AddPrompt([]int32{1, 2, 3, 1, 2, 3})
	p.Draft()
	p.RecordAccepted(1)
	p.Reset()

	if s := p.Stats(); s != (Stats{}) {
		t.Fatalf("stats not cleared after reset: %+v", s)
	}
	if d := p.Draft(); d != nil {
		t.Fatalf("history not cleared after reset, got draft %v", d)
	}
}

func TestDefaultsApplied(t *testing.T) {
	p := NewPromptLookup(0, 0, 0)
	if p.numDraft != DefaultNumDraft || p.ngramSize != DefaultNgramSize || p.minMatches != DefaultMinMatches {
		t.Fatalf("defaults not applied: %d %d %d", p.numDraft, p.ngramSize, p.minMatches)
	}
}

func TestGeneratedTokensExtendHistory(t *testing.T) {
	// No repeat in the prompt, but a generated token creates one, after which a
	// draft becomes available. This mirrors the decode loop feeding tokens back.
	p := NewPromptLookup(4, 2, 1)
	p.AddPrompt([]int32{5, 6, 7})
	if d := p.Draft(); d != nil {
		t.Fatalf("no repeat yet, got %v", d)
	}
	// Append 6 7 again so the window [6 7] recurs.
	p.AddToken(6)
	p.AddToken(7)
	draft := p.Draft()
	if len(draft) == 0 {
		t.Fatal("expected a draft once the window recurred")
	}
}
