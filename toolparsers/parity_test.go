// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"encoding/json"
	"os"
	"testing"
)

// The fixtures below are captured from the reference implementation by
// testdata/parity/toolparsers/gen_fixtures.py. Tool-call ids are random in
// both implementations and are stripped before comparison; everything else
// (names, arguments, content, and the streaming structure) must match exactly.

type callJSON struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type extractJSON struct {
	ToolsCalled bool       `json:"tools_called"`
	ToolCalls   []callJSON `json:"tool_calls"`
	Content     *string    `json:"content"`
}

// fnDelta is a normalized streaming tool-call entry (no id/type).
type fnDelta struct {
	Index     int     `json:"index"`
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

type streamMsg struct {
	Content   *string   `json:"content"`
	ToolCalls []fnDelta `json:"tool_calls"`
}

type toolFixture struct {
	Parser  string         `json:"parser"`
	Label   string         `json:"label"`
	Full    string         `json:"full"`
	Request map[string]any `json:"request"`
	Deltas  []string       `json:"deltas"`
	Extract extractJSON    `json:"extract"`
	Stream  []streamMsg    `json:"stream"`
}

func tpPtrEq(a, b *string) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func tpShow(p *string) string {
	if p == nil {
		return "<nil>"
	}
	return `"` + *p + `"`
}

// normDelta strips ids/types and flattens the function payload so a captured
// stream message can be compared with what the Go parser emits.
func normDelta(d *StreamDelta) streamMsg {
	m := streamMsg{Content: d.Content}
	for _, tc := range d.ToolCalls {
		var name, args *string
		if tc.Function != nil {
			name = tc.Function.Name
			args = tc.Function.Arguments
		}
		m.ToolCalls = append(m.ToolCalls, fnDelta{Index: tc.Index, Name: name, Arguments: args})
	}
	return m
}

func TestToolParserParity(t *testing.T) {
	data, err := os.ReadFile("../testdata/parity/toolparsers/fixtures.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fixtures []toolFixture
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}
	if len(fixtures) == 0 {
		t.Fatal("no fixtures loaded")
	}

	for _, f := range fixtures {
		t.Run(f.Parser+"/"+f.Label, func(t *testing.T) {
			var req Request
			if f.Request != nil {
				req = Request(f.Request)
			}

			// Full extraction.
			p, ok := Get(f.Parser)
			if !ok {
				t.Fatalf("unknown parser %q", f.Parser)
			}
			got := p.ExtractToolCalls(f.Full, req)
			if got.ToolsCalled != f.Extract.ToolsCalled {
				t.Errorf("tools_called: got %v want %v", got.ToolsCalled, f.Extract.ToolsCalled)
			}
			if !tpPtrEq(got.Content, f.Extract.Content) {
				t.Errorf("content: got %s want %s", tpShow(got.Content), tpShow(f.Extract.Content))
			}
			if len(got.ToolCalls) != len(f.Extract.ToolCalls) {
				t.Fatalf("tool-call count: got %d want %d\n  got=%+v", len(got.ToolCalls), len(f.Extract.ToolCalls), got.ToolCalls)
			}
			for i, want := range f.Extract.ToolCalls {
				if got.ToolCalls[i].Name != want.Name {
					t.Errorf("call %d name: got %q want %q", i, got.ToolCalls[i].Name, want.Name)
				}
				if got.ToolCalls[i].Arguments != want.Arguments {
					t.Errorf("call %d arguments:\n  got  %q\n  want %q", i, got.ToolCalls[i].Arguments, want.Arguments)
				}
			}

			// Streaming replay.
			sp, _ := Get(f.Parser)
			sp.Reset()
			var emitted []streamMsg
			prev := ""
			for _, d := range f.Deltas {
				cur := prev + d
				if msg := sp.ExtractToolCallsStreaming(prev, cur, d, req); msg != nil {
					emitted = append(emitted, normDelta(msg))
				}
				prev = cur
			}
			if len(emitted) != len(f.Stream) {
				t.Fatalf("stream chunk count: got %d want %d\n  got=%+v\n  want=%+v", len(emitted), len(f.Stream), emitted, f.Stream)
			}
			for i := range emitted {
				if !tpPtrEq(emitted[i].Content, f.Stream[i].Content) {
					t.Errorf("stream %d content: got %s want %s", i, tpShow(emitted[i].Content), tpShow(f.Stream[i].Content))
				}
				if len(emitted[i].ToolCalls) != len(f.Stream[i].ToolCalls) {
					t.Errorf("stream %d tool-call count: got %d want %d", i, len(emitted[i].ToolCalls), len(f.Stream[i].ToolCalls))
					continue
				}
				for j := range emitted[i].ToolCalls {
					g, w := emitted[i].ToolCalls[j], f.Stream[i].ToolCalls[j]
					if g.Index != w.Index {
						t.Errorf("stream %d call %d index: got %d want %d", i, j, g.Index, w.Index)
					}
					if !tpPtrEq(g.Name, w.Name) {
						t.Errorf("stream %d call %d name: got %s want %s", i, j, tpShow(g.Name), tpShow(w.Name))
					}
					if !tpPtrEq(g.Arguments, w.Arguments) {
						t.Errorf("stream %d call %d arguments: got %s want %s", i, j, tpShow(g.Arguments), tpShow(w.Arguments))
					}
				}
			}
		})
	}
}
