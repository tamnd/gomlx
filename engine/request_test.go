// SPDX-License-Identifier: Apache-2.0

package engine

import "testing"

func TestRequestStatusFinished(t *testing.T) {
	cases := []struct {
		s        RequestStatus
		finished bool
		reason   string
	}{
		{StatusWaiting, false, ""},
		{StatusRunning, false, ""},
		{StatusPreempted, false, ""},
		{StatusFinishedStopped, true, "stop"},
		{StatusFinishedLengthCapped, true, "length"},
		{StatusFinishedAborted, true, "abort"},
	}
	for _, c := range cases {
		if c.s.IsFinished() != c.finished {
			t.Errorf("%v IsFinished: got %v want %v", c.s, c.s.IsFinished(), c.finished)
		}
		if c.s.FinishReason() != c.reason {
			t.Errorf("%v FinishReason: got %q want %q", c.s, c.s.FinishReason(), c.reason)
		}
	}
}

func TestDefaultSamplingParams(t *testing.T) {
	p := DefaultSamplingParams()
	if p.MaxTokens != 256 || p.Temperature != 0.7 || p.TopP != 0.9 {
		t.Errorf("unexpected defaults: %+v", p)
	}
	if p.RepetitionPenalty != 1.0 {
		t.Errorf("repetition penalty should default to 1.0, got %v", p.RepetitionPenalty)
	}
	if p.Stop == nil || p.StopTokenIDs == nil {
		t.Error("stop slices should be non-nil")
	}
}
