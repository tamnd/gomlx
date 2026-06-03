// SPDX-License-Identifier: Apache-2.0

package middleware

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// TestAllowWithinLimit admits requests up to the limit and rejects the next one.
func TestAllowWithinLimit(t *testing.T) {
	now := time.Unix(0, 0)
	l := newRateLimiterClock(3, time.Minute, func() time.Time { return now })
	for i := 0; i < 3; i++ {
		if !l.Allow("a") {
			t.Fatalf("request %d should be allowed", i)
		}
	}
	if l.Allow("a") {
		t.Fatal("fourth request should be rejected")
	}
}

// TestWindowSlides frees capacity once old requests fall outside the window.
func TestWindowSlides(t *testing.T) {
	now := time.Unix(0, 0)
	l := newRateLimiterClock(2, time.Second, func() time.Time { return now })
	if !l.Allow("a") {
		t.Fatal("first should be allowed")
	}
	if !l.Allow("a") {
		t.Fatal("second should be allowed")
	}
	if l.Allow("a") {
		t.Fatal("third within window should be rejected")
	}
	now = now.Add(1100 * time.Millisecond) // both earlier requests now expired
	if !l.Allow("a") {
		t.Fatal("request after window should be allowed again")
	}
}

// TestClientsAreIndependent checks one client's spending does not limit another.
func TestClientsAreIndependent(t *testing.T) {
	now := time.Unix(0, 0)
	l := newRateLimiterClock(1, time.Minute, func() time.Time { return now })
	if !l.Allow("a") || !l.Allow("b") {
		t.Fatal("each client gets its own budget")
	}
	if l.Allow("a") || l.Allow("b") {
		t.Fatal("each client is over its own limit")
	}
}

// TestZeroLimitDisables treats a non-positive limit as no limiting.
func TestZeroLimitDisables(t *testing.T) {
	l := NewRateLimiter(0, time.Minute)
	for i := 0; i < 100; i++ {
		if !l.Allow("a") {
			t.Fatalf("limit<=0 should allow everything, blocked at %d", i)
		}
	}
}

// TestSweepDropsIdleClients reclaims clients whose requests have all expired.
func TestSweepDropsIdleClients(t *testing.T) {
	now := time.Unix(0, 0)
	l := newRateLimiterClock(2, time.Second, func() time.Time { return now })
	l.Allow("a")
	l.Allow("b")
	if l.ActiveClients() != 2 {
		t.Fatalf("active=%d want 2", l.ActiveClients())
	}
	now = now.Add(2 * time.Second)
	l.Sweep()
	if l.ActiveClients() != 0 {
		t.Fatalf("active after sweep=%d want 0", l.ActiveClients())
	}
}

// TestMiddlewareReturns429 checks the HTTP wrapper passes traffic through until
// the limit and then answers 429 with the OpenAI-shaped error body.
func TestMiddlewareReturns429(t *testing.T) {
	now := time.Unix(0, 0)
	l := newRateLimiterClock(1, time.Minute, func() time.Time { return now })
	var hits int
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusOK)
	})
	h := RateLimit(l)(next)

	req := func() *httptest.ResponseRecorder {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		r.Header.Set("Authorization", "Bearer secret")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		return rec
	}

	if rec := req(); rec.Code != http.StatusOK {
		t.Fatalf("first request code=%d want 200", rec.Code)
	}
	rec := req()
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("second request code=%d want 429", rec.Code)
	}
	if hits != 1 {
		t.Fatalf("handler ran %d times, want 1", hits)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content-type=%q", ct)
	}
	if body := rec.Body.String(); body != `{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}` {
		t.Fatalf("unexpected body: %s", body)
	}
}

// TestMiddlewareSkipsPublicAndPreflight checks health checks and CORS preflight
// are never rate limited.
func TestMiddlewareSkipsPublicAndPreflight(t *testing.T) {
	now := time.Unix(0, 0)
	l := newRateLimiterClock(1, time.Minute, func() time.Time { return now })
	h := RateLimit(l)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	for i := 0; i < 5; i++ {
		r := httptest.NewRequest(http.MethodGet, "/health", nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		if rec.Code != http.StatusOK {
			t.Fatalf("health request %d blocked: %d", i, rec.Code)
		}
	}
	for i := 0; i < 5; i++ {
		r := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		if rec.Code != http.StatusOK {
			t.Fatalf("preflight %d blocked: %d", i, rec.Code)
		}
	}
}

// TestConcurrentAllow exercises the lock under the race detector and checks the
// limit is honored exactly across goroutines.
func TestConcurrentAllow(t *testing.T) {
	now := time.Unix(0, 0)
	const limit = 50
	l := newRateLimiterClock(limit, time.Minute, func() time.Time { return now })
	var wg sync.WaitGroup
	var mu sync.Mutex
	allowed := 0
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if l.Allow("a") {
				mu.Lock()
				allowed++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if allowed != limit {
		t.Fatalf("allowed=%d want exactly %d", allowed, limit)
	}
}
