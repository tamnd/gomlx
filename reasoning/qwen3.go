// SPDX-License-Identifier: Apache-2.0

package reasoning

import "strings"

// Qwen3 parses <think>...</think> reasoning for Qwen3 models. The chat
// template injects <think> in the prompt, so a missing </think> means the
// model never finished reasoning (truncated/garbled) rather than "no thought".
type Qwen3 struct {
	baseThinking
}

// NewQwen3 returns a Qwen3 reasoning parser.
func NewQwen3() *Qwen3 {
	return &Qwen3{baseThinking{startToken: "<think>", endToken: "</think>"}}
}

var _ ReasoningParser = (*Qwen3)(nil)

// ExtractReasoning handles the truncated-reasoning case before delegating to
// the shared FSM: with no </think> but a <think> present, everything after
// <think> is reasoning and content is nil.
func (q *Qwen3) ExtractReasoning(text string) (*string, *string) {
	if !strings.Contains(text, q.endToken) {
		if strings.Contains(text, q.startToken) {
			_, reasoning, _ := partition(text, q.startToken)
			return stripNil(reasoning), nil
		}
		return nil, &text
	}
	return q.baseThinking.ExtractReasoning(text)
}

// FinalizeStreaming corrects the no-close-tag case: if </think> never
// appeared, the FSM defaulted everything to reasoning, so re-emit the
// accumulated text as content (dropping a template-injected <think> prefix).
func (q *Qwen3) FinalizeStreaming(accumulated string) *DeltaMessage {
	if strings.Contains(accumulated, q.endToken) {
		return nil
	}
	if accumulated != "" {
		cleaned := strings.TrimPrefix(accumulated, q.startToken)
		if cleaned != "" {
			return &DeltaMessage{Content: &cleaned}
		}
	}
	return nil
}
