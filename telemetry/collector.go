// SPDX-License-Identifier: Apache-2.0

// Package telemetry records what the server is doing under load: how many
// requests it has served, how many tokens it has read and generated, how often
// requests fail, and how long they take. The serving layer calls into a
// Collector from each request goroutine as work completes, and a metrics
// endpoint reads a snapshot back out. Keeping the accounting here, off the
// request handlers, means the handlers stay simple and the numbers are gathered
// in one place that is safe to call from many goroutines at once.
package telemetry

import (
	"slices"
	"sync"
	"time"
)

// RequestStat is the record one finished request contributes. TTFT is the time
// to the first token (zero for non-streaming or failed requests); Latency is the
// wall time for the whole request.
type RequestStat struct {
	PromptTokens     int
	CompletionTokens int
	TTFT             time.Duration
	Latency          time.Duration
	Failed           bool
}

// Snapshot is a point-in-time view of the collected metrics.
type Snapshot struct {
	Requests         uint64
	Failed           uint64
	PromptTokens     uint64
	CompletionTokens uint64
	Uptime           time.Duration

	// GenTokensPerSec is completion tokens divided by uptime, the server's
	// sustained output rate since it started.
	GenTokensPerSec float64

	// Latency and time-to-first-token percentiles over the recent window.
	LatencyP50 time.Duration
	LatencyP90 time.Duration
	TTFTP50    time.Duration
	TTFTP90    time.Duration
}

// Collector aggregates request metrics. It is safe for concurrent use.
type Collector struct {
	now func() time.Time // injectable clock; defaults to time.Now

	mu               sync.Mutex
	start            time.Time
	requests         uint64
	failed           uint64
	promptTokens     uint64
	completionTokens uint64

	// Bounded windows of recent latencies and first-token times, used for
	// percentiles. Older samples are overwritten in a ring so memory is fixed
	// regardless of how long the server runs.
	lat    []time.Duration
	ttft   []time.Duration
	latPos int
	ttPos  int
	window int
}

// New returns a Collector that keeps the most recent window samples for
// percentile estimates. A window of zero or less uses a default of 1024.
func New(window int) *Collector {
	return newWithClock(window, time.Now)
}

func newWithClock(window int, now func() time.Time) *Collector {
	if window <= 0 {
		window = 1024
	}
	return &Collector{
		now:    now,
		start:  now(),
		window: window,
	}
}

// Record adds one finished request to the totals.
func (c *Collector) Record(s RequestStat) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.requests++
	if s.Failed {
		c.failed++
	}
	c.promptTokens += uint64(max0(s.PromptTokens))
	c.completionTokens += uint64(max0(s.CompletionTokens))
	if s.Latency > 0 {
		c.lat = ringPush(c.lat, &c.latPos, c.window, s.Latency)
	}
	if s.TTFT > 0 {
		c.ttft = ringPush(c.ttft, &c.ttPos, c.window, s.TTFT)
	}
}

// Snapshot returns the current aggregate metrics.
func (c *Collector) Snapshot() Snapshot {
	c.mu.Lock()
	defer c.mu.Unlock()
	up := c.now().Sub(c.start)
	var rate float64
	if up > 0 {
		rate = float64(c.completionTokens) / up.Seconds()
	}
	lp50, lp90 := percentiles(c.lat)
	tp50, tp90 := percentiles(c.ttft)
	return Snapshot{
		Requests:         c.requests,
		Failed:           c.failed,
		PromptTokens:     c.promptTokens,
		CompletionTokens: c.completionTokens,
		Uptime:           up,
		GenTokensPerSec:  rate,
		LatencyP50:       lp50,
		LatencyP90:       lp90,
		TTFTP50:          tp50,
		TTFTP90:          tp90,
	}
}

// ringPush appends v to a bounded ring, overwriting the oldest sample once the
// window is full. It returns the (possibly grown) backing slice.
func ringPush(buf []time.Duration, pos *int, window int, v time.Duration) []time.Duration {
	if len(buf) < window {
		return append(buf, v)
	}
	buf[*pos] = v
	*pos = (*pos + 1) % window
	return buf
}

// percentiles returns the p50 and p90 of the samples without disturbing the
// caller's slice. An empty input yields zeroes.
func percentiles(samples []time.Duration) (p50, p90 time.Duration) {
	if len(samples) == 0 {
		return 0, 0
	}
	sorted := make([]time.Duration, len(samples))
	copy(sorted, samples)
	slices.Sort(sorted)
	return quantile(sorted, 0.50), quantile(sorted, 0.90)
}

// quantile returns the value at fraction q of a sorted slice using the
// nearest-rank method.
func quantile(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	rank := int(q * float64(len(sorted)))
	if rank >= len(sorted) {
		rank = len(sorted) - 1
	}
	return sorted[rank]
}

func max0(n int) int {
	if n < 0 {
		return 0
	}
	return n
}
