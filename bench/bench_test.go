// SPDX-License-Identifier: Apache-2.0

package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// fakeServer answers /v1/chat/completions in both streaming and non-streaming
// form so the harness can be exercised without a model.
func fakeServer(tokens int) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Stream bool `json:"stream"`
		}
		_ = decode(r, &req)
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			for range tokens {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
				if fl != nil {
					fl.Flush()
				}
				time.Sleep(time.Millisecond)
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"usage":{"completion_tokens":%d}}`, tokens)
	})
	return httptest.NewServer(mux)
}

func decode(r *http.Request, v any) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(v)
}

func TestRunNonStream(t *testing.T) {
	srv := fakeServer(10)
	defer srv.Close()

	rep, err := Run(context.Background(), Config{
		URL:         srv.URL,
		Model:       "test",
		Prompt:      "hi",
		MaxTokens:   10,
		Concurrency: 4,
		Requests:    20,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.Requests != 20 || rep.Failures != 0 {
		t.Fatalf("requests=%d failures=%d", rep.Requests, rep.Failures)
	}
	if rep.CompletionTokens != 200 {
		t.Errorf("completion tokens: got %d want 200", rep.CompletionTokens)
	}
	if rep.RPS <= 0 || rep.TokensPerSec <= 0 {
		t.Errorf("rates not positive: rps=%v toks=%v", rep.RPS, rep.TokensPerSec)
	}
}

func TestRunStreamTTFT(t *testing.T) {
	srv := fakeServer(5)
	defer srv.Close()

	rep, err := Run(context.Background(), Config{
		URL:         srv.URL,
		Model:       "test",
		Prompt:      "hi",
		MaxTokens:   5,
		Concurrency: 2,
		Requests:    6,
		Stream:      true,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.Failures != 0 {
		t.Fatalf("failures: %d", rep.Failures)
	}
	if rep.CompletionTokens != 30 {
		t.Errorf("completion tokens: got %d want 30", rep.CompletionTokens)
	}
	if rep.TTFTp50 <= 0 {
		t.Errorf("ttft p50 not measured: %v", rep.TTFTp50)
	}
	if rep.TPOT <= 0 {
		t.Errorf("time per token not measured: %v", rep.TPOT)
	}
}

func TestPercentileIndex(t *testing.T) {
	cases := []struct{ n, p, want int }{
		{10, 50, 4},
		{10, 99, 9},
		{1, 99, 0},
		{100, 50, 49},
	}
	for _, c := range cases {
		if got := percentileIndex(c.n, c.p); got != c.want {
			t.Errorf("percentileIndex(%d,%d)=%d want %d", c.n, c.p, got, c.want)
		}
	}
}

func TestFailuresCounted(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	rep, err := Run(context.Background(), Config{URL: srv.URL, Requests: 3, Concurrency: 1})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.Failures != 3 {
		t.Errorf("failures: got %d want 3", rep.Failures)
	}
}
