// SPDX-License-Identifier: Apache-2.0

// Package bench is a load generator for OpenAI-compatible chat endpoints. It
// drives a configurable number of concurrent clients through
// /v1/chat/completions and reports the metrics that matter for serving: time to
// first token, per-token latency, single-stream and aggregate token throughput,
// and requests per second. Because it speaks the OpenAI wire format, the same
// harness measures gomlx and any other server that implements it, which is how
// the two are compared on identical hardware and weights.
package bench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
)

// Config controls a benchmark run.
type Config struct {
	URL         string // base URL, e.g. http://127.0.0.1:8000
	Model       string
	APIKey      string
	Prompt      string
	MaxTokens   int
	Temperature float64
	Concurrency int
	Requests    int  // total requests across all workers
	Stream      bool // measure time to first token via SSE
}

// Result is the measurement for a single request.
type Result struct {
	Start            time.Time
	TTFT             time.Duration // time to first token (streaming only)
	Total            time.Duration
	CompletionTokens int
	Err              error
}

// Report aggregates the results of a run.
type Report struct {
	Config           Config
	Wall             time.Duration
	Requests         int
	Failures         int
	CompletionTokens int

	RPS          float64 // successful requests per second
	TokensPerSec float64 // aggregate completion tokens per second

	TTFTp50 time.Duration
	TTFTp99 time.Duration

	// TPOT is the mean time per output token after the first, a measure of
	// steady-state decode speed.
	TPOT time.Duration

	results []Result
}

// nowFunc is overridable in tests; production uses time.Now.
var nowFunc = time.Now

// Run executes the benchmark and returns the aggregated report.
func Run(ctx context.Context, cfg Config) (Report, error) {
	if cfg.Concurrency < 1 {
		cfg.Concurrency = 1
	}
	if cfg.Requests < 1 {
		cfg.Requests = cfg.Concurrency
	}
	if cfg.MaxTokens < 1 {
		cfg.MaxTokens = 128
	}

	client := &http.Client{}
	jobs := make(chan int)
	results := make([]Result, cfg.Requests)
	var wg sync.WaitGroup

	wallStart := nowFunc()
	for w := 0; w < cfg.Concurrency; w++ {
		wg.Go(func() {
			for i := range jobs {
				results[i] = doRequest(ctx, client, cfg)
			}
		})
	}
	for i := 0; i < cfg.Requests; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	wall := nowFunc().Sub(wallStart)

	return summarize(cfg, results, wall), nil
}

func summarize(cfg Config, results []Result, wall time.Duration) Report {
	r := Report{Config: cfg, Wall: wall, Requests: len(results), results: results}
	var ttfts []time.Duration
	var tpotSum time.Duration
	var tpotN int
	for _, res := range results {
		if res.Err != nil {
			r.Failures++
			continue
		}
		r.CompletionTokens += res.CompletionTokens
		if cfg.Stream && res.TTFT > 0 {
			ttfts = append(ttfts, res.TTFT)
			if res.CompletionTokens > 1 {
				tpotSum += res.Total - res.TTFT
				tpotN += res.CompletionTokens - 1
			}
		}
	}
	secs := wall.Seconds()
	if secs > 0 {
		ok := r.Requests - r.Failures
		r.RPS = float64(ok) / secs
		r.TokensPerSec = float64(r.CompletionTokens) / secs
	}
	if len(ttfts) > 0 {
		slices.Sort(ttfts)
		r.TTFTp50 = ttfts[percentileIndex(len(ttfts), 50)]
		r.TTFTp99 = ttfts[percentileIndex(len(ttfts), 99)]
	}
	if tpotN > 0 {
		r.TPOT = tpotSum / time.Duration(tpotN)
	}
	return r
}

// percentileIndex returns the index into a sorted slice of length n for the
// given percentile.
func percentileIndex(n, p int) int {
	if n == 0 {
		return 0
	}
	idx := (p*n+99)/100 - 1 // nearest-rank: ceil(p/100 * n) - 1
	if idx >= n {
		idx = n - 1
	}
	if idx < 0 {
		idx = 0
	}
	return idx
}

func doRequest(ctx context.Context, client *http.Client, cfg Config) Result {
	res := Result{Start: nowFunc()}

	body := map[string]any{
		"model":       cfg.Model,
		"messages":    []map[string]string{{"role": "user", "content": cfg.Prompt}},
		"max_tokens":  cfg.MaxTokens,
		"temperature": cfg.Temperature,
		"stream":      cfg.Stream,
	}
	raw, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.URL+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		res.Err = err
		return res
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		res.Err = err
		return res
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		res.Err = fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
		return res
	}

	if cfg.Stream {
		res.CompletionTokens, res.TTFT = readStream(resp.Body, res.Start)
	} else {
		res.CompletionTokens = readNonStream(resp.Body)
	}
	res.Total = nowFunc().Sub(res.Start)
	return res
}

// readStream counts SSE content deltas and records the time of the first one.
func readStream(body io.Reader, start time.Time) (tokens int, ttft time.Duration) {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		for _, c := range chunk.Choices {
			if c.Delta.Content != "" {
				if tokens == 0 {
					ttft = nowFunc().Sub(start)
				}
				tokens++
			}
		}
	}
	return tokens, ttft
}

// readNonStream reads usage.completion_tokens from a full response.
func readNonStream(body io.Reader) int {
	var out struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if json.NewDecoder(body).Decode(&out) != nil {
		return 0
	}
	return out.Usage.CompletionTokens
}

// String renders a human-readable summary.
func (r Report) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "requests:        %d (%d failed)\n", r.Requests, r.Failures)
	fmt.Fprintf(&b, "concurrency:     %d\n", r.Config.Concurrency)
	fmt.Fprintf(&b, "wall:            %s\n", r.Wall.Round(time.Millisecond))
	fmt.Fprintf(&b, "throughput:      %.1f tok/s\n", r.TokensPerSec)
	fmt.Fprintf(&b, "requests/sec:    %.2f\n", r.RPS)
	if r.Config.Stream {
		fmt.Fprintf(&b, "ttft p50/p99:    %s / %s\n", r.TTFTp50.Round(time.Millisecond), r.TTFTp99.Round(time.Millisecond))
		fmt.Fprintf(&b, "time/token:      %s\n", r.TPOT.Round(time.Microsecond))
	}
	return b.String()
}
