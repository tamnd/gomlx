// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/tamnd/gomlx/api"
)

func retrieve(d *Deps, id string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models/"+id, nil)
	req.SetPathValue("model_id", id)
	d.RetrieveModel(rec, req)
	return rec
}

func TestRetrieveModelByConfiguredName(t *testing.T) {
	rec := retrieve(&Deps{Model: "qwen3.5-4b"}, "qwen3.5-4b")
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d, want 200", rec.Code)
	}
	var out api.ModelInfo
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.ID != "qwen3.5-4b" || out.Object != "model" || out.OwnedBy != "gomlx" {
		t.Fatalf("model info wrong: %+v", out)
	}
}

func TestRetrieveModelByResolvedPath(t *testing.T) {
	// The server was started with the alias; asking by the resolved hf_path is
	// the same model.
	rec := retrieve(&Deps{Model: "qwen3.5-4b"}, "mlx-community/Qwen3.5-4B-MLX-4bit")
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d, want 200", rec.Code)
	}
}

func TestRetrieveModelByAliasOfSamePath(t *testing.T) {
	// The server was started with the raw path; asking by an alias that resolves
	// to that path is the same model.
	rec := retrieve(&Deps{Model: "mlx-community/Qwen3.5-4B-MLX-4bit"}, "qwen3.5-4b")
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d, want 200", rec.Code)
	}
}

func TestRetrieveModelUnknownIs404(t *testing.T) {
	for _, id := range []string{"gpt-4", "qwen3.5-4b-8bit", ""} {
		rec := retrieve(&Deps{Model: "qwen3.5-4b"}, id)
		if rec.Code != http.StatusNotFound {
			t.Fatalf("id=%q status=%d, want 404", id, rec.Code)
		}
	}
}
