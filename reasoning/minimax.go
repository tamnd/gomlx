// SPDX-License-Identifier: Apache-2.0

package reasoning

import (
	"regexp"
	"strings"
)

// MiniMax parses MiniMax-family output, which thinks inline without explicit
// <think> tags. It buffers the start of the output, heuristically detects a
// reasoning preamble, and finds where the real answer begins. MiniMax does
// sometimes emit <think> tags, which are handled first.
type MiniMax struct {
	buffer        string
	decided       bool
	isReasoning   bool
	transitionPos int
}

// NewMiniMax returns a MiniMax (heuristic, tagless) reasoning parser.
func NewMiniMax() *MiniMax { return &MiniMax{} }

var _ ReasoningParser = (*MiniMax)(nil)

// minimaxBufferSize bounds how much text is buffered before deciding; the
// decision actually fires at min(bufferSize, 80) chars (see streaming).
const minimaxDecideAt = 80

var (
	minimaxStart = regexp.MustCompile(`(?i)^(?:\s*)(?:` +
		`(?:The\s+user\s+(?:asks|wants|is\s+asking|requests|said|query|question))` +
		`|(?:I\s+(?:need\s+to|should|will|can|want\s+to|have\s+to|must|am\s+going\s+to))` +
		`|(?:Let\s+me\s+(?:think|check|analyze|figure|consider|look|read|review|process))` +
		`|(?:This\s+(?:is\s+a|requires|seems|looks\s+like|appears))` +
		`|(?:First,?\s+(?:I|let|we))` +
		`|(?:(?:So|Now|OK|Okay|Alright|Well),?\s+(?:the\s+user|I\s+need|let\s+me|I\s+should))` +
		`|(?:what's\s+worth\s+storing)` +
		`|(?:(?:Analyzing|Thinking|Processing|Considering|Evaluating|Extracting)\s)` +
		`|(?:用户(?:想|要|需要|问|请求|说|希望|让我))` +
		`|(?:我(?:需要|应该|将|可以|要|得|必须))` +
		`|(?:让我(?:想|看|分析|检查|考虑|读|处理|review))` +
		`|(?:这(?:是一个|需要|似乎|看起来|个))` +
		`|(?:首先[，,]?(?:我|让|我们))` +
		`|(?:(?:好的|那么|现在|所以)[，,]?(?:用户|我需要|让我|我应该))` +
		`|(?:(?:分析|思考|处理|考虑|评估|提取)(?:一下|中|着))` +
		`)`)

	minimaxTransition = regexp.MustCompile(`(?im)(?:` +
		`(?:^|\n\n)(?:(?:The\s+)?(?:answer|result|output|response|solution)\s*(?:is|:))` +
		`|(?:^|\n)(?:Thus\s+(?:answer|final|the\s+answer|response)\s*[:\.])` +
		`|(?:^|\n\n)(?:` + "```" + `)` +
		`|(?:^|\n\n)(?:Here\s+(?:is|are)\s)` +
		`|(?:^|\n\n)(?:(?:Sure|Of\s+course|Absolutely)[!,.]?\s)` +
		`|(?:^|\n\n)(?:I'(?:d|ll|m)\s+(?:happy|glad)\s+to\s)` +
		`|(?:<minimax:tool_call>)` +
		`|(?:<tool_call>)` +
		`|(?:<invoke\s)` +
		`|(?:^|\n\n)(?:\d+\.\s+\*\*)` +
		`|(?:^|\n\n)(?:##\s)` +
		`|(?:^|\n\n)\*\*` +
		`|(?:^|\n\n)(?:(?:答案|结果|输出|响应|解决方案)\s*(?:是|：|:))` +
		`|(?:^|\n\n)(?:(?:好的|当然|没问题)[！!，,]?\s)` +
		`|(?:^|\n\n)(?:以下是)` +
		`)`)

	minimaxDirect = regexp.MustCompile(`^(?:\s*)(?:` + "```" +
		`|(?:<minimax:tool_call>)|(?:<tool_call>)|(?:<invoke\s)|(?:#+\s)|(?:\{)|(?:\[))`)
)

// ResetState clears streaming state for a new request.
func (m *MiniMax) ResetState() {
	m.buffer = ""
	m.decided = false
	m.isReasoning = false
	m.transitionPos = 0
}

// ExtractReasoning splits complete output into (reasoning, content).
func (m *MiniMax) ExtractReasoning(out string) (*string, *string) {
	if strings.Contains(out, "<think>") || strings.Contains(out, "</think>") {
		if strings.Contains(out, "</think>") {
			parts := strings.SplitN(out, "</think>", 2)
			reasoning := strings.TrimSpace(strings.ReplaceAll(parts[0], "<think>", ""))
			var content string
			if len(parts) > 1 {
				content = strings.TrimSpace(parts[1])
			}
			return nz(reasoning), nz(content)
		}
	}

	if minimaxDirect.MatchString(out) {
		return nil, &out
	}
	if !minimaxStart.MatchString(out) {
		return nil, &out
	}

	if loc := minimaxTransition.FindStringIndex(out); loc != nil {
		reasoning := strings.TrimSpace(out[:loc[0]])
		content := strings.TrimSpace(out[loc[0]:])
		if len(reasoning) < 10 {
			return nil, &out
		}
		return nz(reasoning), nz(content)
	}

	parts := strings.SplitN(out, "\n\n", 2)
	if len(parts) == 2 {
		first, second := parts[0], parts[1]
		if minimaxStart.MatchString(first) && strings.TrimSpace(second) != "" {
			return nz(strings.TrimSpace(first)), nz(strings.TrimSpace(second))
		}
	}
	return nil, &out
}

// ExtractReasoningStreaming buffers, decides, then routes deltas.
func (m *MiniMax) ExtractReasoningStreaming(prev, cur, delta string) *DeltaMessage {
	if strings.Contains(delta, "</think>") {
		idx := strings.Index(delta, "</think>")
		m.decided = true
		m.isReasoning = false
		return &DeltaMessage{
			Reasoning: nz(delta[:idx]),
			Content:   nz(delta[idx+len("</think>"):]),
		}
	}
	if strings.Contains(delta, "<think>") {
		cleaned := strings.ReplaceAll(delta, "<think>", "")
		m.decided = true
		m.isReasoning = true
		if cleaned != "" {
			return &DeltaMessage{Reasoning: &cleaned}
		}
		return nil
	}

	if m.decided {
		if !m.isReasoning {
			return &DeltaMessage{Content: &delta}
		}
		loc := minimaxTransition.FindStringIndex(cur[m.transitionPos:])
		if loc != nil {
			absPos := m.transitionPos + loc[0]
			m.isReasoning = false
			prevLen := len(prev)
			if absPos >= prevLen {
				reasoning := delta[:absPos-prevLen]
				content := strings.TrimLeft(delta[absPos-prevLen:], "\n")
				return &DeltaMessage{Reasoning: nz(reasoning), Content: nz(content)}
			}
			return &DeltaMessage{Content: &delta}
		}
		return &DeltaMessage{Reasoning: &delta}
	}

	// Still buffering.
	m.buffer = cur
	if len(m.buffer) < minimaxDecideAt {
		if minimaxDirect.MatchString(m.buffer) {
			m.decided = true
			m.isReasoning = false
			return &DeltaMessage{Content: &cur}
		}
		return nil
	}

	m.decided = true
	if !minimaxStart.MatchString(m.buffer) {
		m.isReasoning = false
		return &DeltaMessage{Content: &cur}
	}

	m.isReasoning = true
	if loc := minimaxTransition.FindStringIndex(m.buffer); loc != nil {
		m.isReasoning = false
		absPos := loc[0]
		reasoning := strings.TrimSpace(cur[:absPos])
		content := strings.TrimLeft(cur[absPos:], "\n")
		return &DeltaMessage{Reasoning: nz(reasoning), Content: nz(content)}
	}
	m.transitionPos = max(0, len(m.buffer)-20)
	return &DeltaMessage{Reasoning: &cur}
}

// FinalizeStreaming emits content when the stream never produced any.
func (m *MiniMax) FinalizeStreaming(accumulated string) *DeltaMessage {
	if !m.decided {
		if accumulated != "" {
			return &DeltaMessage{Content: &accumulated}
		}
		return nil
	}
	if !m.isReasoning {
		return nil
	}
	_, content := m.ExtractReasoning(accumulated)
	if content != nil && *content != accumulated {
		return &DeltaMessage{Content: content}
	}
	return &DeltaMessage{Content: &accumulated}
}
