// SPDX-License-Identifier: Apache-2.0

// Package reasoning extracts thinking/reasoning content from model output,
// separating it from the final answer (e.g. <think>...</think>).
package reasoning

import "strings"

// DeltaMessage is one streaming increment. Reasoning and Content should not both
// be set except on the transition chunk. A nil pointer means "field absent".
type DeltaMessage struct {
	Role      *string
	Content   *string
	Reasoning *string
}

// ReasoningParser extracts reasoning from complete or streaming output.
type ReasoningParser interface {
	// ExtractReasoning splits complete output into (reasoning, content); either
	// may be nil.
	ExtractReasoning(modelOutput string) (reasoning, content *string)
	// ExtractReasoningStreaming processes one delta using the
	// previous+delta=current model. Returns nil to skip (e.g. a pure tag).
	ExtractReasoningStreaming(previousText, currentText, deltaText string) *DeltaMessage
	// ResetState clears per-request streaming state.
	ResetState()
	// FinalizeStreaming optionally emits a correction after the stream ends.
	FinalizeStreaming(accumulatedText string) *DeltaMessage
}

// nz returns a pointer to s, or nil if s is empty (mirrors `x if x else None`).
func nz(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// strip trims surrounding whitespace.
func strip(s string) string { return strings.TrimSpace(s) }

// stripNil returns the trimmed string as a pointer, or nil if empty after
// trimming (mirrors `x.strip() or None`).
func stripNil(s string) *string { return nz(strip(s)) }

// baseThinking implements the shared <think>...</think> FSM. Concrete parsers
// embed it and set the tokens; some override ExtractReasoning/streaming.
type baseThinking struct {
	startToken string
	endToken   string
	sawAnyTag  bool
}

// ResetState resets streaming state for a new request.
func (b *baseThinking) ResetState() { b.sawAnyTag = false }

// FinalizeStreaming is a no-op by default.
func (b *baseThinking) FinalizeStreaming(string) *DeltaMessage { return nil }

// ExtractReasoning handles the four complete-output cases.
func (b *baseThinking) ExtractReasoning(text string) (*string, *string) {
	hasStart := strings.Contains(text, b.startToken)
	hasEnd := strings.Contains(text, b.endToken)

	// Case 1: both tags present.
	if hasStart && hasEnd {
		_, afterStart, _ := partition(text, b.startToken)
		reasoning, content, _ := partition(afterStart, b.endToken)
		return stripNil(reasoning), stripNil(content)
	}
	// Case 2: only closing tag (think injected in the prompt).
	if hasEnd {
		reasoning, content, _ := partition(text, b.endToken)
		return stripNil(reasoning), stripNil(content)
	}
	// Case 3: only start tag (incomplete reasoning).
	if hasStart {
		_, reasoning, _ := partition(text, b.startToken)
		return stripNil(reasoning), nil
	}
	// Case 4: no tags, pure content.
	return nil, &text
}

// ExtractReasoningStreaming handles incremental deltas.
func (b *baseThinking) ExtractReasoningStreaming(prev, cur, delta string) *DeltaMessage {
	sd := strip(delta)
	if sd == b.startToken || sd == b.endToken {
		return nil
	}
	startInPrev := strings.Contains(prev, b.startToken)
	startInCur := strings.Contains(cur, b.startToken)
	endInPrev := strings.Contains(prev, b.endToken)
	endInDelta := strings.Contains(delta, b.endToken)

	// Case 1: explicit <think> in the text.
	if startInCur {
		b.sawAnyTag = true
		return b.handleExplicit(delta, startInPrev, endInPrev, endInDelta)
	}
	// Case 2: only </think>, implicit reasoning mode.
	if strings.Contains(cur, b.endToken) {
		b.sawAnyTag = true
		return b.handleImplicit(delta, endInPrev, endInDelta)
	}
	// Case 3: no tags yet, treat as reasoning and correct later if needed.
	return &DeltaMessage{Reasoning: &delta}
}

func (b *baseThinking) handleExplicit(delta string, startInPrev, endInPrev, endInDelta bool) *DeltaMessage {
	startInDelta := strings.Contains(delta, b.startToken)

	if startInPrev {
		if endInDelta {
			idx := strings.Index(delta, b.endToken)
			return &DeltaMessage{
				Reasoning: nz(delta[:idx]),
				Content:   nz(delta[idx+len(b.endToken):]),
			}
		}
		if endInPrev {
			return &DeltaMessage{Content: &delta}
		}
		return &DeltaMessage{Reasoning: &delta}
	}

	if startInDelta {
		startIdx := strings.Index(delta, b.startToken)
		if endInDelta {
			endIdx := strings.Index(delta, b.endToken)
			return &DeltaMessage{
				Reasoning: nz(delta[startIdx+len(b.startToken) : endIdx]),
				Content:   nz(delta[endIdx+len(b.endToken):]),
			}
		}
		return &DeltaMessage{Reasoning: nz(delta[startIdx+len(b.startToken):])}
	}

	return &DeltaMessage{Content: &delta}
}

func (b *baseThinking) handleImplicit(delta string, endInPrev, endInDelta bool) *DeltaMessage {
	if endInDelta {
		idx := strings.Index(delta, b.endToken)
		return &DeltaMessage{
			Reasoning: nz(delta[:idx]),
			Content:   nz(delta[idx+len(b.endToken):]),
		}
	}
	if endInPrev {
		return &DeltaMessage{Content: &delta}
	}
	return &DeltaMessage{Reasoning: &delta}
}

// partition mirrors Python's str.partition: returns (before, after, found).
func partition(s, sep string) (string, string, bool) {
	before, after, found := strings.Cut(s, sep)
	return before, after, found
}
