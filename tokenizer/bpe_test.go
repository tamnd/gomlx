// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"reflect"
	"testing"
)

// tinyTokenizer is a hand-built byte-level BPE with just enough vocab and
// merges to exercise the encoder, the ignore_merges short circuit, and special
// tokens, without needing the multi-megabyte real tokenizer.json in CI.
const tinyTokenizer = `{
  "added_tokens": [{"id": 100, "content": "<|end|>"}],
  "model": {
    "type": "BPE",
    "ignore_merges": true,
    "vocab": {"a": 0, "b": 1, "Ġ": 2, "ab": 3, "Ġab": 4, "Ġa": 5},
    "merges": [["a", "b"], ["Ġ", "a"], ["Ġa", "b"]]
  }
}`

func mustTiny(t *testing.T) *Tokenizer {
	t.Helper()
	tk, err := FromJSON([]byte(tinyTokenizer))
	if err != nil {
		t.Fatalf("FromJSON: %v", err)
	}
	return tk
}

func TestEncodeMergesAndSpace(t *testing.T) {
	tk := mustTiny(t)
	// "ab ab" splits into "ab" and " ab"; byte-level makes the space a Ġ, and
	// both pieces are whole vocab entries, so ignore_merges returns them direct.
	got := tk.Encode("ab ab")
	want := []int{3, 4}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("encode: got %v want %v", got, want)
	}
}

func TestEncodeWithoutIgnoreMerges(t *testing.T) {
	// Force the merge path. The pair (a,b) has the lowest rank, so the two byte
	// symbols merge into "ab" rather than short-circuiting on the whole word.
	tk := mustTiny(t)
	tk.ignoreMerges = false
	got := tk.Encode("ab")
	want := []int{3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("encode: got %v want %v", got, want)
	}
}

func TestSpecialTokenSplit(t *testing.T) {
	tk := mustTiny(t)
	got := tk.Encode("ab<|end|>ab")
	want := []int{3, 100, 3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("encode: got %v want %v", got, want)
	}
	if d := tk.Decode(want); d != "abab" {
		t.Errorf("decode (skip special): got %q want %q", d, "abab")
	}
	if d := tk.DecodeWithSpecial(want); d != "ab<|end|>ab" {
		t.Errorf("decode (keep special): got %q want %q", d, "ab<|end|>ab")
	}
}

func TestPreTokenize(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"ab ab", []string{"ab", " ab"}},
		{"12", []string{"1", "2"}},         // digits split one at a time
		{"a  b", []string{"a", " ", " b"}}, // run of spaces: leading space joins "b"
		{"hi!\n", []string{"hi", "!\n"}},   // symbol run absorbs trailing newline
		{"don't", []string{"don", "'t"}},   // contraction
		{"  ", []string{"  "}},             // trailing-only whitespace stays whole
	}
	for _, c := range cases {
		got := preTokenize(c.in)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("preTokenize(%q): got %v want %v", c.in, got, c.want)
		}
	}
}

func TestParseMergeStringForm(t *testing.T) {
	a, b, err := parseMerge([]byte(`"foo bar"`))
	if err != nil || a != "foo" || b != "bar" {
		t.Fatalf("string merge: got (%q,%q,%v)", a, b, err)
	}
	a, b, err = parseMerge([]byte(`["x","y"]`))
	if err != nil || a != "x" || b != "y" {
		t.Fatalf("array merge: got (%q,%q,%v)", a, b, err)
	}
}
