// SPDX-License-Identifier: Apache-2.0

package reasoning

import (
	"encoding/json"
	"os"
	"testing"
)

// msgJSON mirrors a captured DeltaMessage (reasoning/content only). Pointers
// distinguish JSON null (absent) from "" (empty string), matching Python None
// vs "".
type msgJSON struct {
	Reasoning *string `json:"reasoning"`
	Content   *string `json:"content"`
}

type fixture struct {
	Parser   string    `json:"parser"`
	Label    string    `json:"label"`
	Full     string    `json:"full"`
	Deltas   []string  `json:"deltas"`
	Extract  msgJSON   `json:"extract"`
	Stream   []msgJSON `json:"stream"`
	Finalize *msgJSON  `json:"finalize"`
}

func ptrEq(a, b *string) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func show(p *string) string {
	if p == nil {
		return "<nil>"
	}
	return `"` + *p + `"`
}

// TestReasoningParity replays golden fixtures
// (testdata/parity/reasoning/gen_fixtures.py) and asserts the parsers produce
// identical reasoning/content splits for both the full extraction and the
// chunk-by-chunk streaming path.
func TestReasoningParity(t *testing.T) {
	data, err := os.ReadFile("../testdata/parity/reasoning/fixtures.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fixtures []fixture
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}
	if len(fixtures) == 0 {
		t.Fatal("no fixtures loaded")
	}

	for _, f := range fixtures {
		t.Run(f.Parser+"/"+f.Label, func(t *testing.T) {
			// Full extraction.
			p, err := Get(f.Parser)
			if err != nil {
				t.Fatalf("Get(%q): %v", f.Parser, err)
			}
			gotR, gotC := p.ExtractReasoning(f.Full)
			if !ptrEq(gotR, f.Extract.Reasoning) || !ptrEq(gotC, f.Extract.Content) {
				t.Errorf("ExtractReasoning mismatch\n  reasoning: got %s want %s\n  content:   got %s want %s",
					show(gotR), show(f.Extract.Reasoning), show(gotC), show(f.Extract.Content))
			}

			// Streaming.
			sp, _ := Get(f.Parser)
			sp.ResetState()
			var emitted []msgJSON
			prev := ""
			for _, d := range f.Deltas {
				cur := prev + d
				if msg := sp.ExtractReasoningStreaming(prev, cur, d); msg != nil {
					emitted = append(emitted, msgJSON{Reasoning: msg.Reasoning, Content: msg.Content})
				}
				prev = cur
			}
			if len(emitted) != len(f.Stream) {
				t.Fatalf("stream chunk count: got %d want %d\n  got=%v", len(emitted), len(f.Stream), emitted)
			}
			for i := range emitted {
				if !ptrEq(emitted[i].Reasoning, f.Stream[i].Reasoning) || !ptrEq(emitted[i].Content, f.Stream[i].Content) {
					t.Errorf("stream chunk %d mismatch\n  reasoning: got %s want %s\n  content:   got %s want %s",
						i, show(emitted[i].Reasoning), show(f.Stream[i].Reasoning),
						show(emitted[i].Content), show(f.Stream[i].Content))
				}
			}

			// Finalize.
			fin := sp.FinalizeStreaming(prev)
			switch {
			case fin == nil && f.Finalize == nil:
				// match
			case fin == nil || f.Finalize == nil:
				t.Errorf("finalize presence mismatch: got %v want %v", fin, f.Finalize)
			default:
				if !ptrEq(fin.Reasoning, f.Finalize.Reasoning) || !ptrEq(fin.Content, f.Finalize.Content) {
					t.Errorf("finalize mismatch\n  reasoning: got %s want %s\n  content:   got %s want %s",
						show(fin.Reasoning), show(f.Finalize.Reasoning),
						show(fin.Content), show(f.Finalize.Content))
				}
			}
		})
	}
}
