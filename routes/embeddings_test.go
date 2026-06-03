// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/gomlx/api"
)

// mockEmbedder is a deterministic, GPU-free Embedder for exercising the route.
// Its vectors carry no semantic meaning; they are stable functions of the input
// length so the encoding, truncation, and usage paths can be checked.
type mockEmbedder struct {
	name string
	dim  int
}

func (m *mockEmbedder) ModelName() string { return m.name }

func (m *mockEmbedder) CountTokens(texts []string) int {
	n := 0
	for _, t := range texts {
		n += len(strings.Fields(t))
	}
	return n
}

func (m *mockEmbedder) vector(seed int) []float32 {
	v := make([]float32, m.dim)
	for j := range v {
		v[j] = float32((seed+j)%7) + 1
	}
	return v
}

func (m *mockEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = m.vector(len(t))
	}
	return out, nil
}

func (m *mockEmbedder) EmbedTokens(_ context.Context, batches [][]int) ([][]float32, error) {
	out := make([][]float32, len(batches))
	for i, b := range batches {
		out[i] = m.vector(len(b))
	}
	return out, nil
}

func embedDeps() *Deps {
	return &Deps{Embedder: &mockEmbedder{name: "embed-model", dim: 8}}
}

// embedRespRaw decodes a response while keeping each embedding as raw JSON so a
// test can tell a float array apart from a base64 string.
type embedRespRaw struct {
	Object string `json:"object"`
	Model  string `json:"model"`
	Data   []struct {
		Object    string          `json:"object"`
		Index     int             `json:"index"`
		Embedding json.RawMessage `json:"embedding"`
	} `json:"data"`
	Usage api.EmbeddingUsage `json:"usage"`
}

func callEmbeddings(d *Deps, body string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	d.Embeddings(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body)))
	return rec
}

func TestEmbeddingsUnconfigured(t *testing.T) {
	rec := callEmbeddings(&Deps{}, `{"model":"x","input":"hello"}`)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d, want 503", rec.Code)
	}
}

func TestEmbeddingsModelMismatch(t *testing.T) {
	rec := callEmbeddings(embedDeps(), `{"model":"other-model","input":"hello"}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d, want 400", rec.Code)
	}
}

func TestEmbeddingsSingleString(t *testing.T) {
	rec := callEmbeddings(embedDeps(), `{"model":"embed-model","input":"hello world"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body)
	}
	var out embedRespRaw
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "list" || out.Model != "embed-model" {
		t.Fatalf("envelope wrong: %+v", out)
	}
	if len(out.Data) != 1 || out.Data[0].Object != "embedding" || out.Data[0].Index != 0 {
		t.Fatalf("data wrong: %+v", out.Data)
	}
	var vec []float32
	if err := json.Unmarshal(out.Data[0].Embedding, &vec); err != nil {
		t.Fatalf("embedding is not a float array: %v", err)
	}
	if len(vec) != 8 {
		t.Fatalf("dim=%d, want 8", len(vec))
	}
	// "hello world" is two whitespace tokens.
	if out.Usage.PromptTokens != 2 || out.Usage.TotalTokens != 2 {
		t.Fatalf("usage=%+v, want 2/2", out.Usage)
	}
}

func TestEmbeddingsListOfStrings(t *testing.T) {
	rec := callEmbeddings(embedDeps(), `{"model":"embed-model","input":["a","b","c"]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body)
	}
	var out embedRespRaw
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if len(out.Data) != 3 {
		t.Fatalf("data len=%d, want 3", len(out.Data))
	}
}

func TestEmbeddingsSingleTokenList(t *testing.T) {
	rec := callEmbeddings(embedDeps(), `{"model":"embed-model","input":[101,202,303]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body)
	}
	var out embedRespRaw
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if len(out.Data) != 1 {
		t.Fatalf("a single token list is one input, got %d", len(out.Data))
	}
	if out.Usage.PromptTokens != 3 {
		t.Fatalf("prompt_tokens=%d, want 3", out.Usage.PromptTokens)
	}
}

func TestEmbeddingsBatchTokenLists(t *testing.T) {
	rec := callEmbeddings(embedDeps(), `{"model":"embed-model","input":[[1,2],[3,4,5]]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body)
	}
	var out embedRespRaw
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if len(out.Data) != 2 {
		t.Fatalf("data len=%d, want 2", len(out.Data))
	}
	if out.Usage.PromptTokens != 5 {
		t.Fatalf("prompt_tokens=%d, want 5", out.Usage.PromptTokens)
	}
}

func TestEmbeddingsRejectsBadInput(t *testing.T) {
	cases := map[string]string{
		"empty array":         `{"model":"embed-model","input":[]}`,
		"empty token seq":     `{"model":"embed-model","input":[[1,2],[]]}`,
		"single empty tokens": `{"model":"embed-model","input":[]}`,
		"mixed array":         `{"model":"embed-model","input":["a",1]}`,
		"dimensions zero":     `{"model":"embed-model","input":"x","dimensions":0}`,
		"malformed body":      `{"model":`,
		"missing input":       `{"model":"embed-model"}`,
		"object input":        `{"model":"embed-model","input":{"a":1}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			rec := callEmbeddings(embedDeps(), body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status=%d, want 400 (body=%s)", rec.Code, rec.Body)
			}
		})
	}
}

func TestEmbeddingsDimensionsTruncateAndRenorm(t *testing.T) {
	rec := callEmbeddings(embedDeps(), `{"model":"embed-model","input":"hello","dimensions":4}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body)
	}
	var out embedRespRaw
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	var vec []float64
	if err := json.Unmarshal(out.Data[0].Embedding, &vec); err != nil {
		t.Fatal(err)
	}
	if len(vec) != 4 {
		t.Fatalf("dim=%d, want 4", len(vec))
	}
	var sum float64
	for _, x := range vec {
		sum += x * x
	}
	if math.Abs(math.Sqrt(sum)-1) > 1e-5 {
		t.Fatalf("truncated vector is not unit norm: |v|=%v", math.Sqrt(sum))
	}
}

func TestEmbeddingsDimensionsTooLarge(t *testing.T) {
	rec := callEmbeddings(embedDeps(), `{"model":"embed-model","input":"hello","dimensions":99}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d, want 400", rec.Code)
	}
}

func TestEmbeddingsBase64Encoding(t *testing.T) {
	rec := callEmbeddings(embedDeps(), `{"model":"embed-model","input":"hello","encoding_format":"base64"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body)
	}
	var out embedRespRaw
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	var encoded string
	if err := json.Unmarshal(out.Data[0].Embedding, &encoded); err != nil {
		t.Fatalf("base64 embedding should be a JSON string: %v", err)
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("not valid base64: %v", err)
	}
	if len(raw) != 8*4 {
		t.Fatalf("decoded %d bytes, want 32 (8 float32)", len(raw))
	}
}
