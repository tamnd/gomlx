// SPDX-License-Identifier: Apache-2.0

package reasoning

import "strings"

// noTagContentThreshold is the character count past which no-tag output is
// treated as content rather than reasoning. Real reasoning models emit <think>
// within the first few tokens; ~64 chars (15-20 tokens) is a safe cutoff.
const noTagContentThreshold = 64

// DeepSeekR1 parses <think>...</think> reasoning, more leniently than Qwen3:
// the opening <think> is often implicit, so a lone </think> means everything
// before it is reasoning.
type DeepSeekR1 struct {
	baseThinking
}

// NewDeepSeekR1 returns a DeepSeek-R1 reasoning parser.
func NewDeepSeekR1() *DeepSeekR1 {
	return &DeepSeekR1{baseThinking{startToken: "<think>", endToken: "</think>"}}
}

var _ ReasoningParser = (*DeepSeekR1)(nil)

// ExtractReasoning handles the implicit-start case (only </think> present)
// before delegating to the shared FSM.
func (d *DeepSeekR1) ExtractReasoning(text string) (*string, *string) {
	hasStart := strings.Contains(text, d.startToken)
	hasEnd := strings.Contains(text, d.endToken)
	if hasEnd && !hasStart {
		reasoning, content, _ := partition(text, d.endToken)
		return stripNil(reasoning), stripNil(content)
	}
	if !hasEnd && !hasStart {
		return nil, &text
	}
	return d.baseThinking.ExtractReasoning(text)
}

// ExtractReasoningStreaming adds two behaviours on top of the shared FSM: a
// length threshold for no-tag output, and a split when </think> arrives in a
// delta without <think> ever having been seen.
func (d *DeepSeekR1) ExtractReasoningStreaming(prev, cur, delta string) *DeltaMessage {
	hasTags := strings.Contains(cur, d.startToken) || strings.Contains(cur, d.endToken)
	if !hasTags && !d.sawAnyTag && len(cur) >= noTagContentThreshold {
		return &DeltaMessage{Content: &delta}
	}

	result := d.baseThinking.ExtractReasoningStreaming(prev, cur, delta)

	if result != nil {
		startInPrev := strings.Contains(prev, d.startToken)
		startInDelta := strings.Contains(delta, d.startToken)
		endInDelta := strings.Contains(delta, d.endToken)
		if !startInPrev && !startInDelta && endInDelta {
			idx := strings.Index(delta, d.endToken)
			return &DeltaMessage{
				Reasoning: nz(delta[:idx]),
				Content:   nz(delta[idx+len(d.endToken):]),
			}
		}
	}
	return result
}

// FinalizeStreaming reclassifies short no-tag output (which the FSM treated as
// reasoning) as content.
func (d *DeepSeekR1) FinalizeStreaming(accumulated string) *DeltaMessage {
	if !d.sawAnyTag && accumulated != "" && len(accumulated) < noTagContentThreshold {
		return &DeltaMessage{Content: &accumulated}
	}
	return nil
}
