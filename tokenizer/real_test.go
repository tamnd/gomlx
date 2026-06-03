// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestRealQwen3Fixtures checks the encoder and decoder against ids captured from
// the reference Qwen3 tokenizer. The 11 MB tokenizer.json is not committed, so
// the test loads it from a local model directory and skips when it is absent,
// which is the case on CI. Point GOMLX_QWEN3_TOKENIZER at a tokenizer.json to
// run it elsewhere.
func TestRealQwen3Fixtures(t *testing.T) {
	path := os.Getenv("GOMLX_QWEN3_TOKENIZER")
	if path == "" {
		path = filepath.Join(os.Getenv("HOME"), "models", "qwen3-0.6b", "tokenizer.json")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("reference tokenizer not present at %s", path)
	}
	tk, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join("testdata", "qwen3_fixtures.json"))
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fixtures []struct {
		Text    string `json:"text"`
		IDs     []int  `json:"ids"`
		Decoded string `json:"decoded"`
	}
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}

	for _, f := range fixtures {
		if got := tk.Encode(f.Text); !reflect.DeepEqual(got, f.IDs) {
			t.Errorf("encode %q\n got=%v\nwant=%v", f.Text, got, f.IDs)
		}
		if got := tk.Decode(f.IDs); got != f.Decoded {
			t.Errorf("decode %q: got %q want %q", f.Text, got, f.Decoded)
		}
	}
}
