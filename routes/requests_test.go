// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/gomlx/engine"
)

func TestCancelRegistry(t *testing.T) {
	var reg cancelRegistry

	ctx, cancel := context.WithCancel(context.Background())
	reg.add("a", cancel)

	if !reg.cancel("a") {
		t.Fatal("cancel of a registered id should report true")
	}
	select {
	case <-ctx.Done():
	default:
		t.Fatal("cancel should have fired the context's cancel func")
	}
	if reg.cancel("a") {
		t.Fatal("a cancelled id should no longer be in flight")
	}
	if reg.cancel("missing") {
		t.Fatal("cancel of an unknown id should report false")
	}
}

func TestCancelRegistryRemoveDoesNotCancel(t *testing.T) {
	var reg cancelRegistry
	fired := false
	reg.add("b", func() { fired = true })
	reg.remove("b")
	if reg.cancel("b") {
		t.Fatal("a removed id should not be in flight")
	}
	if fired {
		t.Fatal("remove should forget the request without cancelling it")
	}
}

func cancelReq(d *Deps, id string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/requests/"+id+"/cancel", nil)
	req.SetPathValue("request_id", id)
	d.CancelRequest(rec, req)
	return rec
}

func TestCancelRequestInFlight(t *testing.T) {
	d := &Deps{Model: "m"}
	fired := false
	d.Cancels.add("req-1", func() { fired = true })

	rec := cancelReq(d, "req-1")
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d, want 200", rec.Code)
	}
	if !fired {
		t.Fatal("cancelling an in-flight request should fire its cancel func")
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out["cancelled"] != true || out["id"] != "req-1" || out["model"] != "m" {
		t.Fatalf("response body wrong: %+v", out)
	}
}

func TestCancelRequestUnknownIs404(t *testing.T) {
	d := &Deps{Model: "m"}
	for _, id := range []string{"nope", ""} {
		if rec := cancelReq(d, id); rec.Code != http.StatusNotFound {
			t.Fatalf("id=%q status=%d, want 404", id, rec.Code)
		}
	}
}

// TestStreamRegistersAndCleansUp proves the id the client sees in the stream is
// the same id the cancel registry tracks, and that a finished stream is
// deregistered: cancelling its id afterwards is a 404.
func TestStreamRegistersAndCleansUp(t *testing.T) {
	d := &Deps{Engine: engine.NewMockEngine("m"), Model: "m"}
	rec := httptest.NewRecorder()
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	d.ChatCompletions(rec, req)

	id := firstStreamID(t, rec.Body.String())
	if !strings.HasPrefix(id, "chatcmpl-") {
		t.Fatalf("stream id %q is not a chat completion id", id)
	}
	// The stream ran to completion synchronously, so its id must already be
	// deregistered.
	if rec := cancelReq(d, id); rec.Code != http.StatusNotFound {
		t.Fatalf("a finished stream should be deregistered, got status %d", rec.Code)
	}
}

// firstStreamID pulls the response id out of the first SSE data chunk.
func firstStreamID(t *testing.T, sse string) string {
	t.Helper()
	for _, line := range strings.Split(sse, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk struct {
			ID string `json:"id"`
		}
		if json.Unmarshal([]byte(data), &chunk) == nil && chunk.ID != "" {
			return chunk.ID
		}
	}
	t.Fatalf("no id found in stream: %q", sse)
	return ""
}
