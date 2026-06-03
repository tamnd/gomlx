// SPDX-License-Identifier: Apache-2.0

package telemetry

import (
	"sync"
	"testing"
	"time"
)

func TestRecordTotals(t *testing.T) {
	c := New(0)
	c.Record(RequestStat{PromptTokens: 10, CompletionTokens: 5})
	c.Record(RequestStat{PromptTokens: 3, CompletionTokens: 7, Failed: true})
	s := c.Snapshot()
	if s.Requests != 2 || s.Failed != 1 {
		t.Fatalf("requests=%d failed=%d, want 2 and 1", s.Requests, s.Failed)
	}
	if s.PromptTokens != 13 || s.CompletionTokens != 12 {
		t.Fatalf("tokens: prompt=%d completion=%d", s.PromptTokens, s.CompletionTokens)
	}
}

func TestThroughputUsesInjectedClock(t *testing.T) {
	now := time.Unix(0, 0)
	c := newWithClock(0, func() time.Time { return now })
	c.Record(RequestStat{CompletionTokens: 100})
	now = now.Add(2 * time.Second)
	s := c.Snapshot()
	if s.Uptime != 2*time.Second {
		t.Fatalf("uptime=%v want 2s", s.Uptime)
	}
	if s.GenTokensPerSec != 50 {
		t.Fatalf("rate=%v want 50 tok/s", s.GenTokensPerSec)
	}
}

func TestLatencyPercentiles(t *testing.T) {
	c := New(0)
	// Latencies 1..10 ms; p50 (nearest-rank, rank=5) = 6ms, p90 (rank=9) = 10ms.
	for i := 1; i <= 10; i++ {
		c.Record(RequestStat{Latency: time.Duration(i) * time.Millisecond})
	}
	s := c.Snapshot()
	if s.LatencyP50 != 6*time.Millisecond {
		t.Errorf("p50=%v want 6ms", s.LatencyP50)
	}
	if s.LatencyP90 != 10*time.Millisecond {
		t.Errorf("p90=%v want 10ms", s.LatencyP90)
	}
}

func TestTTFTRecordedSeparately(t *testing.T) {
	c := New(0)
	c.Record(RequestStat{TTFT: 5 * time.Millisecond, Latency: 50 * time.Millisecond})
	s := c.Snapshot()
	if s.TTFTP50 != 5*time.Millisecond {
		t.Fatalf("ttft p50=%v want 5ms", s.TTFTP50)
	}
	if s.LatencyP50 != 50*time.Millisecond {
		t.Fatalf("latency p50=%v want 50ms", s.LatencyP50)
	}
}

// TestWindowBoundsMemory checks the ring keeps only the most recent window
// samples and that percentiles reflect the retained window.
func TestWindowBoundsMemory(t *testing.T) {
	c := New(4)
	// Record 1..8 ms; only the last four (5,6,7,8) are retained.
	for i := 1; i <= 8; i++ {
		c.Record(RequestStat{Latency: time.Duration(i) * time.Millisecond})
	}
	if len(c.lat) != 4 {
		t.Fatalf("retained %d samples, want 4", len(c.lat))
	}
	s := c.Snapshot()
	// Sorted window {5,6,7,8}: p50 rank=2 -> 7ms, p90 rank=3 -> 8ms.
	if s.LatencyP50 != 7*time.Millisecond || s.LatencyP90 != 8*time.Millisecond {
		t.Fatalf("windowed percentiles: p50=%v p90=%v", s.LatencyP50, s.LatencyP90)
	}
}

// TestConcurrentRecord exercises the lock under many goroutines; the race
// detector and the final count are the assertions.
func TestConcurrentRecord(t *testing.T) {
	c := New(0)
	var wg sync.WaitGroup
	const n = 50
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.Record(RequestStat{PromptTokens: 1, CompletionTokens: 2, Latency: time.Millisecond})
		}()
	}
	wg.Wait()
	s := c.Snapshot()
	if s.Requests != n || s.CompletionTokens != 2*n {
		t.Fatalf("requests=%d completion=%d", s.Requests, s.CompletionTokens)
	}
}
