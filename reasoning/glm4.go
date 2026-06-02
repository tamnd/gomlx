// SPDX-License-Identifier: Apache-2.0

package reasoning

import "strings"

const (
	glmBoxStart = "<|begin_of_box|>"
	glmBoxEnd   = "<|end_of_box|>"
)

// GLM4 parses <think>...</think> reasoning for the GLM-4 family. Unlike Qwen3,
// GLM-4's template does NOT inject <think>, so output with no tags is genuine
// content (not implicit reasoning). GLM-4.6V also wraps content in
// <|begin_of_box|>...<|end_of_box|> markers, which are stripped.
type GLM4 struct {
	baseThinking
}

// NewGLM4 returns a GLM-4 reasoning parser.
func NewGLM4() *GLM4 {
	return &GLM4{baseThinking{startToken: "<think>", endToken: "</think>"}}
}

var _ ReasoningParser = (*GLM4)(nil)

func glmStripBox(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, glmBoxStart, ""), glmBoxEnd, "")
}

// ExtractReasoning strips box markers before delegating to the shared FSM.
func (g *GLM4) ExtractReasoning(text string) (*string, *string) {
	return g.baseThinking.ExtractReasoning(glmStripBox(text))
}

// ExtractReasoningStreaming strips box markers, then diverges from the shared
// FSM on exactly one branch: pre-tag output is content, not reasoning.
func (g *GLM4) ExtractReasoningStreaming(prev, cur, delta string) *DeltaMessage {
	if strings.Contains(delta, glmBoxStart) || strings.Contains(delta, glmBoxEnd) {
		delta = glmStripBox(delta)
		if delta == "" {
			return nil
		}
		prev = glmStripBox(prev)
		cur = glmStripBox(cur)
	}

	hasTags := strings.Contains(cur, g.startToken) || strings.Contains(cur, g.endToken)
	if !hasTags && !g.sawAnyTag {
		return &DeltaMessage{Content: &delta}
	}
	return g.baseThinking.ExtractReasoningStreaming(prev, cur, delta)
}
