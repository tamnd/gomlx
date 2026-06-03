// SPDX-License-Identifier: Apache-2.0

package speculative

import (
	"slices"
	"testing"
)

func TestSuffixDraftFromRepeat(t *testing.T) {
	d := NewSuffixDecoder(4, 2, 0.3, 0)
	// [1 2 3 1 2 3]: the suffix [2 3] last appeared followed by 1 2 3.
	d.AddPrompt([]int32{1, 2, 3, 1, 2, 3})
	if got := d.Draft(); !slices.Equal(got, []int32{1, 2, 3}) {
		t.Fatalf("draft=%v want [1 2 3]", got)
	}
}

func TestSuffixTruncatesBelowConfidence(t *testing.T) {
	// [5 7] is followed by 8 once and 9 once: a tie. With a confidence floor of
	// 0.6 neither wins, so no draft is proposed.
	d := NewSuffixDecoder(4, 2, 0.6, 0)
	d.AddPrompt([]int32{5, 7, 8, 5, 7, 9, 5, 7})
	if got := d.Draft(); got != nil {
		t.Fatalf("ambiguous continuation should yield no draft, got %v", got)
	}
	// Lowering the floor lets the tie resolve (toward the smaller token id) and
	// a draft is produced.
	e := NewSuffixDecoder(4, 2, 0.3, 0)
	e.AddPrompt([]int32{5, 7, 8, 5, 7, 9, 5, 7})
	if got := e.Draft(); len(got) == 0 || got[0] != 8 {
		t.Fatalf("draft=%v want it to start at 8", got)
	}
}

func TestSuffixPrefersLongestMatch(t *testing.T) {
	// [3] alone is ambiguous (followed by 4 and by 8), but the longer suffix
	// [2 3] appears only before 4, so the longer match decides the draft.
	d := NewSuffixDecoder(4, 2, 0.3, 0)
	d.AddPrompt([]int32{2, 3, 4, 7, 3, 8, 2, 3})
	got := d.Draft()
	if len(got) == 0 || got[0] != 4 {
		t.Fatalf("draft=%v want it to start at 4 from the longer suffix", got)
	}
}

func TestSuffixDraftCappedByMaxDraft(t *testing.T) {
	d := NewSuffixDecoder(2, 2, 0.3, 0) // at most two draft tokens
	d.AddPrompt([]int32{1, 2, 3, 4, 5, 1, 2})
	got := d.Draft()
	if len(got) != 2 {
		t.Fatalf("draft length=%d want 2 (capped)", len(got))
	}
	if !slices.Equal(got, []int32{3, 4}) {
		t.Fatalf("draft=%v want [3 4]", got)
	}
}

func TestSuffixHistoryTrimKeepsIndexConsistent(t *testing.T) {
	// A tight history cap forces trimming on nearly every add. The index must
	// still produce a correct draft from the surviving window and never a stale
	// one from a dropped position.
	d := NewSuffixDecoder(4, 2, 0.3, 4)
	d.AddPrompt([]int32{1, 2, 1, 2, 1, 2})
	if got := d.Draft(); !slices.Equal(got, []int32{1, 2}) {
		t.Fatalf("draft after trim=%v want [1 2]", got)
	}
}

func TestSuffixNoMatchYieldsNil(t *testing.T) {
	d := NewSuffixDecoder(4, 4, 0.3, 0)
	d.AddPrompt([]int32{1, 2, 3, 4, 5})
	if got := d.Draft(); got != nil {
		t.Fatalf("unique tokens should yield no draft, got %v", got)
	}
	empty := NewSuffixDecoder(4, 4, 0.3, 0)
	if got := empty.Draft(); got != nil {
		t.Fatalf("empty history should yield no draft, got %v", got)
	}
}

func TestSuffixStats(t *testing.T) {
	d := NewSuffixDecoder(4, 2, 0.3, 0)
	d.AddPrompt([]int32{1, 2, 3, 1, 2, 3})
	draft := d.Draft()
	d.RecordAccepted(2)

	s := d.Stats()
	if s.NStepCalls != 1 || s.NDraftsReturned != 1 || s.TotalDraftsProposed != 1 {
		t.Fatalf("counts wrong: %+v", s)
	}
	if s.TotalDraftTokensProposed != len(draft) || s.SumDraftLength != len(draft) {
		t.Fatalf("proposed tokens=%d want %d", s.TotalDraftTokensProposed, len(draft))
	}
	if s.TotalDraftTokensAccepted != 2 {
		t.Fatalf("accepted=%d want 2", s.TotalDraftTokensAccepted)
	}
	if got := s.AcceptanceRate(); got != 2.0/float64(len(draft)) {
		t.Fatalf("acceptance rate=%v", got)
	}
	if got := s.MeanAcceptedPerStep(); got != 2.0 {
		t.Fatalf("mean accepted per step=%v want 2", got)
	}
}

func TestSuffixReset(t *testing.T) {
	d := NewSuffixDecoder(4, 2, 0.3, 0)
	d.AddPrompt([]int32{1, 2, 1, 2})
	d.Draft()
	d.RecordAccepted(1)
	d.Reset()
	if s := d.Stats(); s != (DraftStats{}) {
		t.Fatalf("stats not cleared: %+v", s)
	}
	if got := d.Draft(); got != nil {
		t.Fatalf("history not cleared, got %v", got)
	}
}

func TestSuffixDefaultsApplied(t *testing.T) {
	d := NewSuffixDecoder(0, 0, -1, 0)
	if d.maxDraftTokens != DefaultMaxDraftTokens || d.maxSuffixLen != DefaultMaxSuffixLen {
		t.Fatalf("size defaults not applied: %d %d", d.maxDraftTokens, d.maxSuffixLen)
	}
	if d.minConfidence != DefaultMinConfidence {
		t.Fatalf("confidence default not applied: %v", d.minConfidence)
	}
}
