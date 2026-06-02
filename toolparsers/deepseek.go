// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

const (
	deepseekCallsStart = "<｜tool▁calls▁begin｜>"
	deepseekCallsEnd   = "<｜tool▁calls▁end｜>"
	deepseekCallEnd    = "<｜tool▁call▁end｜>"
)

// DeepSeek V3/R1 wrap calls in unicode markers, with each call carrying a
// fenced ```json block. The full pattern includes a type field before the
// separator; the simple pattern omits it.
var (
	deepseekCallRE = regexp.MustCompile(
		"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)\n```json\n(.*?)\n```<｜tool▁call▁end｜>")
	deepseekSimpleRE = regexp.MustCompile(
		"(?s)<｜tool▁call▁begin｜>(.*?)\n```json\n(.*?)\n```<｜tool▁call▁end｜>")
)

// DeepSeekParser handles DeepSeek V3 and R1 tool calls.
type DeepSeekParser struct{ base }

func (p *DeepSeekParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	if !strings.Contains(modelOutput, deepseekCallsStart) {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	var content *string
	if idx := strings.Index(modelOutput, deepseekCallsStart); idx > 0 {
		content = strNil(strip(modelOutput[:idx]))
	}

	var calls []ToolCall
	for _, m := range deepseekCallRE.FindAllStringSubmatch(modelOutput, -1) {
		// m[1] is the type, m[2] the name, m[3] the raw arguments.
		calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[2]), Arguments: strip(m[3])})
	}

	// Fall back to the type-less pattern only when the full one found nothing.
	if len(calls) == 0 {
		for _, m := range deepseekSimpleRE.FindAllStringSubmatch(modelOutput, -1) {
			calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: strip(m[2])})
		}
	}

	if len(calls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: content}
}

func (p *DeepSeekParser) ExtractToolCallsStreaming(_, currentText, deltaText string, req Request) *StreamDelta {
	if !strings.Contains(currentText, deepseekCallsStart) {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	if strings.Contains(deltaText, deepseekCallEnd) || strings.Contains(deltaText, deepseekCallsEnd) {
		return emitAllStreaming(p.ExtractToolCalls(currentText, req))
	}
	return nil
}
