// SPDX-License-Identifier: Apache-2.0

package middleware

import (
	"net"
	"net/http"
	"sync"
	"time"
)

// RateLimiter caps how many requests a single client may make within a rolling
// time window. It keeps, per client, the timestamps of the requests still inside
// the window and admits a new one only while that count is below the limit. The
// window slides continuously rather than resetting on a fixed boundary, so a
// client cannot send a full window's worth of requests at the very end of one
// period and again at the start of the next.
//
// RateLimiter is safe for concurrent use; the HTTP middleware calls it from
// every request goroutine.
type RateLimiter struct {
	limit  int
	window time.Duration
	now    func() time.Time

	mu      sync.Mutex
	clients map[string][]time.Time
}

// NewRateLimiter returns a limiter that allows at most limit requests per client
// within window. A limit of zero or less disables limiting (every request is
// allowed).
func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	return newRateLimiterClock(limit, window, time.Now)
}

func newRateLimiterClock(limit int, window time.Duration, now func() time.Time) *RateLimiter {
	return &RateLimiter{
		limit:   limit,
		window:  window,
		now:     now,
		clients: make(map[string][]time.Time),
	}
}

// Allow reports whether a request from key may proceed, recording it if so.
func (l *RateLimiter) Allow(key string) bool {
	if l.limit <= 0 {
		return true
	}
	now := l.now()
	cutoff := now.Add(-l.window)

	l.mu.Lock()
	defer l.mu.Unlock()
	kept := prune(l.clients[key], cutoff)
	if len(kept) >= l.limit {
		l.clients[key] = kept
		return false
	}
	l.clients[key] = append(kept, now)
	return true
}

// Sweep drops clients with no requests left inside the window, so the map does
// not grow without bound as clients come and go. The server can call this
// periodically; it is also exercised directly by tests.
func (l *RateLimiter) Sweep() {
	cutoff := l.now().Add(-l.window)
	l.mu.Lock()
	defer l.mu.Unlock()
	for key, ts := range l.clients {
		if kept := prune(ts, cutoff); len(kept) == 0 {
			delete(l.clients, key)
		} else {
			l.clients[key] = kept
		}
	}
}

// ActiveClients reports how many clients currently have requests tracked, mainly
// for tests and telemetry.
func (l *RateLimiter) ActiveClients() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.clients)
}

// prune returns the timestamps at or after cutoff. The input is assumed to be in
// ascending order, which it is because Allow only ever appends the current time.
func prune(ts []time.Time, cutoff time.Time) []time.Time {
	i := 0
	for i < len(ts) && !ts[i].After(cutoff) {
		i++
	}
	if i == 0 {
		return ts
	}
	kept := make([]time.Time, len(ts)-i)
	copy(kept, ts[i:])
	return kept
}

// RateLimit wraps a handler so requests over the limit get a 429. The client is
// identified by its bearer token when present, falling back to the remote IP, so
// limiting follows the API key rather than the connection when one is in use.
func RateLimit(l *RateLimiter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodOptions || isPublicPath(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}
			if !l.Allow(clientKey(r)) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}`))
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// clientKey identifies the caller for rate limiting: the bearer token if one is
// sent, otherwise the request's remote IP.
func clientKey(r *http.Request) string {
	if tok := bearerToken(r.Header.Get("Authorization")); tok != "" {
		return "key:" + tok
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	return "ip:" + host
}
