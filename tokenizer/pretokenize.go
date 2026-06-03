// SPDX-License-Identifier: Apache-2.0

package tokenizer

import "unicode"

// preTokenize reproduces the GPT-style pre-tokenization regex that Qwen3 uses:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)
//	|[^\r\n\p{L}\p{N}]?\p{L}+
//	|\p{N}
//	| ?[^\s\p{L}\p{N}]+[\r\n]*
//	|\s*[\r\n]+
//	|\s+(?!\S)
//	|\s+
//
// Go's regexp engine (RE2) cannot express the trailing (?!\S) lookahead, so the
// alternatives are matched by hand. At each position the alternatives are tried
// in order and the first that matches wins, which is exactly how an ordered
// alternation behaves under leftmost-first matching.
func preTokenize(s string) []string {
	r := []rune(s)
	n := len(r)
	var out []string
	i := 0
	for i < n {
		if m := matchPiece(r, i); m > i {
			out = append(out, string(r[i:m]))
			i = m
			continue
		}
		// matchPiece always advances on any input, but guard against a stall.
		out = append(out, string(r[i:i+1]))
		i++
	}
	return out
}

func matchPiece(r []rune, i int) int {
	n := len(r)

	// 1: contractions, case-insensitive.
	if r[i] == '\'' && i+1 < n {
		for _, suf := range contractions {
			if hasFoldSuffix(r, i+1, suf) {
				return i + 1 + len(suf)
			}
		}
	}

	// 2: optional leading non-letter/non-digit, then a run of letters.
	if k := matchLetters(r, i); k > i {
		return k
	}

	// 3: a single number rune.
	if isNumber(r[i]) {
		return i + 1
	}

	// 4: optional leading space, then a run of symbols, then trailing newlines.
	if k := matchSymbols(r, i); k > i {
		return k
	}

	// 5: optional whitespace ending in a run of newlines.
	if k := matchNewlineRun(r, i); k > i {
		return k
	}

	// 6 and 7: whitespace runs. Trailing whitespace before a non-space leaves
	// the last space for the following token (the (?!\S) case); otherwise the
	// whole run matches.
	if isSpace(r[i]) {
		e := i
		for e < n && isSpace(r[e]) {
			e++
		}
		if e == n {
			return e // run reaches the end: take all of it
		}
		if e-1 > i {
			return e - 1 // leave the final space for the next token
		}
		return e // single trailing space before a non-space: catch-all \s+
	}

	return i
}

var contractions = []string{"s", "t", "re", "ve", "m", "ll", "d"}

// hasFoldSuffix reports whether r starting at i case-insensitively equals suf.
func hasFoldSuffix(r []rune, i int, suf string) bool {
	sr := []rune(suf)
	if i+len(sr) > len(r) {
		return false
	}
	for j, c := range sr {
		if unicode.ToLower(r[i+j]) != c {
			return false
		}
	}
	return true
}

// matchLetters handles [^\r\n\p{L}\p{N}]? \p{L}+ : an optional single rune that
// is not a newline, letter, or number, followed by one or more letters.
func matchLetters(r []rune, i int) int {
	n := len(r)
	k := i
	if k < n && r[k] != '\r' && r[k] != '\n' && !isLetter(r[k]) && !isNumber(r[k]) {
		k++
	}
	start := k
	for k < n && isLetter(r[k]) {
		k++
	}
	if k > start {
		return k
	}
	return i
}

// matchSymbols handles  ?[^\s\p{L}\p{N}]+[\r\n]* : an optional leading space, a
// run of symbol runes, then any trailing newlines.
func matchSymbols(r []rune, i int) int {
	n := len(r)
	k := i
	if k < n && r[k] == ' ' {
		k++
	}
	start := k
	for k < n && !isSpace(r[k]) && !isLetter(r[k]) && !isNumber(r[k]) {
		k++
	}
	if k == start {
		return i
	}
	for k < n && (r[k] == '\r' || r[k] == '\n') {
		k++
	}
	return k
}

// matchNewlineRun handles \s*[\r\n]+ : whitespace that ends in newlines. The
// match runs through the last newline of the whitespace block; any plain spaces
// after it are left for a later alternative.
func matchNewlineRun(r []rune, i int) int {
	n := len(r)
	if !isSpace(r[i]) {
		return i
	}
	e := i
	last := -1
	for e < n && isSpace(r[e]) {
		if r[e] == '\r' || r[e] == '\n' {
			last = e
		}
		e++
	}
	if last < 0 {
		return i
	}
	return last + 1
}

func isLetter(c rune) bool { return unicode.IsLetter(c) }
func isNumber(c rune) bool { return unicode.IsNumber(c) }
func isSpace(c rune) bool  { return unicode.IsSpace(c) }
