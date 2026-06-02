// SPDX-License-Identifier: Apache-2.0

package reasoning

// Thinking is the concrete parser for models that wrap reasoning in a
// configurable start/end tag pair (e.g. <think>...</think>). It is the Go
// analogue of BaseThinkingReasoningParser: the shared FSM lives in
// baseThinking, and family-specific parsers embed Thinking and override only
// the methods that diverge.
type Thinking struct {
	baseThinking
}

// NewThinking returns a generic thinking parser for the given tag pair.
func NewThinking(start, end string) *Thinking {
	return &Thinking{baseThinking{startToken: start, endToken: end}}
}

var _ ReasoningParser = (*Thinking)(nil)
