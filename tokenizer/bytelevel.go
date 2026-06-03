// SPDX-License-Identifier: Apache-2.0

// Package tokenizer is a pure-Go reader for Hugging Face byte-level BPE
// tokenizers (the tokenizer.json format used by Qwen3 and the GPT-2 family). It
// has no cgo dependency, so it builds and runs everywhere the rest of the module
// does. The implementation covers the pieces those tokenizers actually use: a
// byte-level alphabet, the GPT-style pre-tokenization split, merge-ranked BPE,
// and recognition of the added/special tokens that appear in chat prompts.
package tokenizer

// byteToRune is the GPT-2 byte-to-unicode table. Byte-level BPE maps every one
// of the 256 byte values to a printable rune so the BPE merges operate on text
// that never contains control characters. Printable ASCII and a band of Latin-1
// map to themselves; the rest are lifted into the U+0100 range.
func byteToRune() ([256]rune, map[rune]byte) {
	var direct []int
	for i := '!'; i <= '~'; i++ {
		direct = append(direct, int(i))
	}
	for i := '¡'; i <= '¬'; i++ {
		direct = append(direct, int(i))
	}
	for i := '®'; i <= 'ÿ'; i++ {
		direct = append(direct, int(i))
	}

	inSet := make(map[int]bool, len(direct))
	for _, b := range direct {
		inSet[b] = true
	}

	var fwd [256]rune
	for _, b := range direct {
		fwd[b] = rune(b)
	}
	n := 0
	for b := 0; b < 256; b++ {
		if !inSet[b] {
			fwd[b] = rune(256 + n)
			n++
		}
	}

	rev := make(map[rune]byte, 256)
	for b := 0; b < 256; b++ {
		rev[fwd[b]] = byte(b)
	}
	return fwd, rev
}

// encodeBytes maps a raw byte string to its byte-level rune string.
func (t *Tokenizer) encodeBytes(s string) string {
	out := make([]rune, 0, len(s))
	for i := 0; i < len(s); i++ {
		out = append(out, t.fwd[s[i]])
	}
	return string(out)
}
