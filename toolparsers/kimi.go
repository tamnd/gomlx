// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

const (
	kimiSectionStart    = "<|tool_calls_section_begin|>"
	kimiSectionStartAlt = "<|tool_call_section_begin|>"
	kimiCallStart       = "<|tool_call_begin|>"
	kimiCallEnd         = "<|tool_call_end|>"
)

// kimiCallRE matches one call: a function id (with an optional :index suffix)
// then the argument blob up to the call-end marker.
var kimiCallRE = regexp.MustCompile(
	`(?s)<\|tool_call_begin\|>\s*([^<]+?)(?::\d+)?\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>`)

// KimiParser handles Kimi K2 and Moonshot section-delimited tool calls.
type KimiParser struct{ base }

func (p *KimiParser) hasSection(text string) bool {
	return strings.Contains(text, kimiSectionStart) ||
		strings.Contains(text, kimiSectionStartAlt) ||
		strings.Contains(text, kimiCallStart)
}

func (p *KimiParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	if !p.hasSection(modelOutput) {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	// Content is whatever precedes the section-start marker.
	var content *string
	for _, marker := range []string{kimiSectionStart, kimiSectionStartAlt} {
		if idx := strings.Index(modelOutput, marker); idx >= 0 {
			if idx > 0 {
				content = strNil(strip(modelOutput[:idx]))
			}
			break
		}
	}

	var calls []ToolCall
	for _, m := range kimiCallRE.FindAllStringSubmatch(modelOutput, -1) {
		funcID, funcArgs := m[1], m[2]
		// func_id may look like "functions.get_weather:0"; recover the bare name.
		name := funcID
		if strings.Contains(name, ":") {
			parts := strings.Split(name, ":")
			name = parts[len(parts)-2]
		}
		dotParts := strings.Split(name, ".")
		name = dotParts[len(dotParts)-1]
		calls = append(calls, ToolCall{ID: genToolID(), Name: strip(name), Arguments: strip(funcArgs)})
	}

	if len(calls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: content}
}

func (p *KimiParser) ExtractToolCallsStreaming(_, currentText, deltaText string, req Request) *StreamDelta {
	if !p.hasSection(currentText) {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	if strings.Contains(deltaText, kimiCallEnd) {
		return emitAllStreaming(p.ExtractToolCalls(currentText, req))
	}
	return nil
}

func (p *KimiParser) HasPendingToolCall(text string) bool {
	return p.hasSection(text) || hasTextFormatToolCall(text)
}
