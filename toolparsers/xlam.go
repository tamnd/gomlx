// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// xLAM emits a JSON array of calls, optionally inside a code fence, after a
// [TOOL_CALLS] tag, or after a </think> block.
var (
	xlamCodeBlockRE = regexp.MustCompile("(?s)```(?:json)?\\s*(.*?)```")
	xlamThinkRE     = regexp.MustCompile(`(?s)</think>\s*(.*)`)
	xlamTagRE       = regexp.MustCompile(`(?s)\[TOOL_CALLS\](.*?)(?:\n|$)`)
)

// XLAMParser handles Salesforce xLAM raw-JSON tool calls.
type XLAMParser struct{ base }

// tryExtractJSON returns the leftover content and the parsed call array. The ok
// result is false when no JSON array could be found.
func (p *XLAMParser) tryExtractJSON(text string) (content *string, calls []any, ok bool) {
	// Code fences first, then the [TOOL_CALLS] tag.
	for _, re := range []*regexp.Regexp{xlamCodeBlockRE, xlamTagRE} {
		for _, m := range re.FindAllStringSubmatch(text, -1) {
			decoded, err := decodeOrdered(strip(m[1]))
			if err != nil {
				continue
			}
			if arr, ok := decoded.([]any); ok {
				stripped := strip(re.ReplaceAllString(text, ""))
				return strNil(stripped), arr, true
			}
		}
	}

	// After a </think> block.
	if m := xlamThinkRE.FindStringSubmatchIndex(text); m != nil {
		after := strip(text[m[2]:m[3]])
		if decoded, err := decodeOrdered(after); err == nil {
			if arr, ok := decoded.([]any); ok {
				cut := strip(text[:m[0]+len("</think>")])
				return strNil(cut), arr, true
			}
		}
	}

	// The whole text as a JSON array.
	trimmed := strip(text)
	if strings.HasPrefix(trimmed, "[") {
		if decoded, err := decodeOrdered(trimmed); err == nil {
			if arr, ok := decoded.([]any); ok {
				return nil, arr, true
			}
		}
	}

	return ptr(text), nil, false
}

func (p *XLAMParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	content, data, ok := p.tryExtractJSON(modelOutput)
	if !ok || len(data) == 0 {
		if content != nil {
			return ExtractedToolCalls{Content: content}
		}
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	var calls []ToolCall
	for _, item := range data {
		call, ok := item.(*omap)
		if !ok || !call.has("name") {
			continue
		}
		name, _ := call.get("name").(string)
		var args any
		if call.has("arguments") {
			args = call.get("arguments")
		} else if call.has("parameters") {
			args = call.get("parameters")
		} else {
			args = newOmap()
		}
		calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(args, false)})
	}

	if len(calls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: content}
}

func (p *XLAMParser) ExtractToolCallsStreaming(_, currentText, deltaText string, req Request) *StreamDelta {
	stripped := strip(currentText)
	hasMarker := strings.Contains(currentText, "```") ||
		strings.Contains(currentText, "[TOOL_CALLS]") ||
		strings.Contains(currentText, "</think>") ||
		(strings.HasPrefix(stripped, "[") && strings.Contains(stripped, "{"))

	if !hasMarker {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	if strings.Contains(deltaText, "]") || strings.Contains(deltaText, "```") {
		return emitAllStreaming(p.ExtractToolCalls(currentText, req))
	}
	return nil
}

func (p *XLAMParser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, "<tool_call>") || hasTextFormatToolCall(text)
}
