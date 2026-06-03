// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
)

// Tokenizer encodes text to token ids and back for a byte-level BPE model.
type Tokenizer struct {
	fwd          [256]rune     // byte -> byte-level rune
	rev          map[rune]byte // byte-level rune -> byte
	vocab        map[string]int
	idToTok      map[int]string
	rank         map[string]int // "a\x00b" -> merge priority, lower binds first
	ignoreMerges bool

	specialByContent map[string]int
	specialByID      map[int]string
	specialRE        *regexp.Regexp // matches any added-token content, longest first
}

// on-disk shapes we read from tokenizer.json.
type tokenizerFile struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Model struct {
		Vocab        map[string]int    `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
		IgnoreMerges bool              `json:"ignore_merges"`
	} `json:"model"`
}

// Load reads a tokenizer.json file.
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return FromJSON(data)
}

// FromJSON builds a tokenizer from the bytes of a tokenizer.json file.
func FromJSON(data []byte) (*Tokenizer, error) {
	var f tokenizerFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("tokenizer: parse json: %w", err)
	}
	if len(f.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocab")
	}

	fwd, rev := byteToRune()
	t := &Tokenizer{
		fwd:              fwd,
		rev:              rev,
		vocab:            f.Model.Vocab,
		idToTok:          make(map[int]string, len(f.Model.Vocab)),
		rank:             make(map[string]int, len(f.Model.Merges)),
		ignoreMerges:     f.Model.IgnoreMerges,
		specialByContent: map[string]int{},
		specialByID:      map[int]string{},
	}
	for tok, id := range f.Model.Vocab {
		t.idToTok[id] = tok
	}

	for i, raw := range f.Model.Merges {
		a, b, err := parseMerge(raw)
		if err != nil {
			return nil, fmt.Errorf("tokenizer: merge %d: %w", i, err)
		}
		t.rank[a+"\x00"+b] = i
	}

	var contents []string
	for _, a := range f.AddedTokens {
		t.specialByContent[a.Content] = a.ID
		t.specialByID[a.ID] = a.Content
		t.idToTok[a.ID] = a.Content
		contents = append(contents, a.Content)
	}
	if len(contents) > 0 {
		// Longest content first so overlapping markers match greedily.
		sort.Slice(contents, func(i, j int) bool { return len(contents[i]) > len(contents[j]) })
		quoted := make([]string, len(contents))
		for i, c := range contents {
			quoted[i] = regexp.QuoteMeta(c)
		}
		t.specialRE = regexp.MustCompile(strings.Join(quoted, "|"))
	}

	return t, nil
}

// parseMerge accepts both merge encodings: a two-element array ["a","b"] and a
// single space-joined string "a b".
func parseMerge(raw json.RawMessage) (string, string, error) {
	var pair []string
	if err := json.Unmarshal(raw, &pair); err == nil {
		if len(pair) != 2 {
			return "", "", fmt.Errorf("array merge has %d parts", len(pair))
		}
		return pair[0], pair[1], nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", "", err
	}
	i := strings.IndexByte(s, ' ')
	if i < 0 {
		return "", "", fmt.Errorf("string merge %q has no space", s)
	}
	return s[:i], s[i+1:], nil
}

// TokenToID returns the id of a token string, or -1 if absent.
func (t *Tokenizer) TokenToID(tok string) int {
	if id, ok := t.vocab[tok]; ok {
		return id
	}
	if id, ok := t.specialByContent[tok]; ok {
		return id
	}
	return -1
}

// Encode turns text into token ids. Added/special token contents that appear in
// the text are recognized and emitted as their ids; the spans between them go
// through pre-tokenization and BPE.
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	t.eachSegment(text, func(seg string, special int) {
		if special >= 0 {
			ids = append(ids, special)
			return
		}
		for _, piece := range preTokenize(seg) {
			ids = append(ids, t.bpe(t.encodeBytes(piece))...)
		}
	})
	return ids
}

// eachSegment walks the text, calling fn for each special-token hit (with its
// id) and each plain span between hits (with special == -1).
func (t *Tokenizer) eachSegment(text string, fn func(seg string, special int)) {
	if t.specialRE == nil {
		fn(text, -1)
		return
	}
	last := 0
	for _, loc := range t.specialRE.FindAllStringIndex(text, -1) {
		if loc[0] > last {
			fn(text[last:loc[0]], -1)
		}
		fn(text[loc[0]:loc[1]], t.specialByContent[text[loc[0]:loc[1]]])
		last = loc[1]
	}
	if last < len(text) {
		fn(text[last:], -1)
	}
}

// bpe applies merge-ranked byte-pair encoding to one byte-level piece and
// returns the resulting token ids.
func (t *Tokenizer) bpe(piece string) []int {
	if piece == "" {
		return nil
	}
	if t.ignoreMerges {
		if id, ok := t.vocab[piece]; ok {
			return []int{id}
		}
	}

	symbols := strings.Split(piece, "")
	for len(symbols) > 1 {
		bestRank := int(^uint(0) >> 1)
		bestAt := -1
		for i := 0; i+1 < len(symbols); i++ {
			if rk, ok := t.rank[symbols[i]+"\x00"+symbols[i+1]]; ok && rk < bestRank {
				bestRank = rk
				bestAt = i
			}
		}
		if bestAt < 0 {
			break
		}
		merged := symbols[bestAt] + symbols[bestAt+1]
		symbols = append(symbols[:bestAt], append([]string{merged}, symbols[bestAt+2:]...)...)
	}

	ids := make([]int, 0, len(symbols))
	for _, s := range symbols {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// Decode turns token ids back into text, dropping special tokens. This matches
// the default behavior of the reference tokenizer, where control markers like
// the chat delimiters never appear in the decoded string.
func (t *Tokenizer) Decode(ids []int) string {
	return t.decode(ids, true)
}

// DecodeWithSpecial is like Decode but keeps special-token contents in the
// output, which is useful when inspecting a raw token stream.
func (t *Tokenizer) DecodeWithSpecial(ids []int) string {
	return t.decode(ids, false)
}

func (t *Tokenizer) decode(ids []int, skipSpecial bool) string {
	var sb strings.Builder
	var pending []rune
	flush := func() {
		if len(pending) == 0 {
			return
		}
		buf := make([]byte, 0, len(pending))
		for _, c := range pending {
			if b, ok := t.rev[c]; ok {
				buf = append(buf, b)
			} else {
				buf = append(buf, []byte(string(c))...)
			}
		}
		sb.Write(buf)
		pending = pending[:0]
	}
	for _, id := range ids {
		if content, ok := t.specialByID[id]; ok {
			flush()
			if !skipSpecial {
				sb.WriteString(content)
			}
			continue
		}
		tok, ok := t.idToTok[id]
		if !ok {
			continue
		}
		pending = append(pending, []rune(tok)...)
	}
	flush()
	return sb.String()
}
