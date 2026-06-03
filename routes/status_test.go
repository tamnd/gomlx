// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func getStatus(d *Deps) statusResponse {
	rec := httptest.NewRecorder()
	d.Status(rec, httptest.NewRequest(http.MethodGet, "/v1/status", nil))
	var out statusResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return out
}

func TestStatusIdle(t *testing.T) {
	out := getStatus(&Deps{Model: "m"})
	if out.Status != "idle" || out.Model != "m" || out.NumRunning != 0 {
		t.Fatalf("idle status wrong: %+v", out)
	}
}

func TestStatusGenerating(t *testing.T) {
	d := &Deps{Model: "m"}
	d.Cancels.add("req-1", func() {})
	d.Cancels.add("req-2", func() {})

	out := getStatus(d)
	if out.Status != "generating" || out.NumRunning != 2 {
		t.Fatalf("generating status wrong: %+v", out)
	}

	// Cancelling frees the slot, so the next snapshot reflects the lower count.
	d.Cancels.cancel("req-1")
	if out := getStatus(d); out.NumRunning != 1 {
		t.Fatalf("num_running=%d after one cancel, want 1", out.NumRunning)
	}
}
