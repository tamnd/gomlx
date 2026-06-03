// SPDX-License-Identifier: Apache-2.0

package cloudrouter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func ptrF(f float64) *float64 { return &f }
func ptrI(i int) *int         { return &i }

func TestShouldRoute(t *testing.T) {
	r := New(Config{Threshold: 1000})
	if r.ShouldRoute(1000) {
		t.Fatal("a request at the threshold should stay local")
	}
	if !r.ShouldRoute(1001) {
		t.Fatal("a request over the threshold should route")
	}

	off := New(Config{Threshold: 0})
	if off.ShouldRoute(1 << 30) {
		t.Fatal("a non-positive threshold should keep everything local")
	}
}

func TestBuildBodyOmitsUnsetParams(t *testing.T) {
	r := New(Config{Model: "cloud/m"})
	body := r.buildBody([]Message{{Role: "user", Content: "hi"}}, Params{
		Temperature: ptrF(0.7),
		MaxTokens:   ptrI(64),
	})

	if body["model"] != "cloud/m" {
		t.Fatalf("model=%v", body["model"])
	}
	if body["temperature"] != 0.7 || body["max_tokens"] != 64 {
		t.Fatalf("set params missing: %v", body)
	}
	for _, k := range []string{"top_p", "frequency_penalty", "presence_penalty", "stop", "tools", "tool_choice", "response_format"} {
		if _, ok := body[k]; ok {
			t.Fatalf("unset param %q should be omitted, body=%v", k, body)
		}
	}
}

func TestCompleteForwardsAndReturnsBody(t *testing.T) {
	var gotPath, gotAuth, gotCT string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		gotPath = req.URL.Path
		gotAuth = req.Header.Get("Authorization")
		gotCT = req.Header.Get("Content-Type")
		raw, _ := io.ReadAll(req.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-x","choices":[]}`))
	}))
	defer srv.Close()

	r := New(Config{Model: "cloud/m", BaseURL: srv.URL, APIKey: "secret", Client: srv.Client()})
	out, err := r.Complete(context.Background(), []Message{{Role: "user", Content: "hi"}}, Params{MaxTokens: ptrI(8)})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if gotPath != "/chat/completions" {
		t.Fatalf("path=%q", gotPath)
	}
	if gotAuth != "Bearer secret" {
		t.Fatalf("auth=%q", gotAuth)
	}
	if gotCT != "application/json" {
		t.Fatalf("content-type=%q", gotCT)
	}
	if gotBody["model"] != "cloud/m" || gotBody["stream"] != false {
		t.Fatalf("forwarded body wrong: %v", gotBody)
	}
	if string(out) != `{"id":"chatcmpl-x","choices":[]}` {
		t.Fatalf("returned body=%s", out)
	}
}

func TestCompleteTrimsTrailingSlashInBaseURL(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		gotPath = req.URL.Path
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	r := New(Config{Model: "m", BaseURL: srv.URL + "/", Client: srv.Client()})
	if _, err := r.Complete(context.Background(), nil, Params{}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if gotPath != "/chat/completions" {
		t.Fatalf("path=%q want /chat/completions (no double slash)", gotPath)
	}
}

func TestCompleteSurfacesUpstreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"slow down"}`))
	}))
	defer srv.Close()

	r := New(Config{Model: "m", BaseURL: srv.URL, Client: srv.Client()})
	_, err := r.Complete(context.Background(), nil, Params{})
	if err == nil {
		t.Fatal("a non-2xx upstream response should be an error")
	}
}

func TestCompleteOmitsAuthWhenNoKey(t *testing.T) {
	var hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		_, hadAuth = req.Header["Authorization"]
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	r := New(Config{Model: "m", BaseURL: srv.URL, Client: srv.Client()})
	if _, err := r.Complete(context.Background(), nil, Params{}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if hadAuth {
		t.Fatal("no API key should mean no Authorization header")
	}
}
