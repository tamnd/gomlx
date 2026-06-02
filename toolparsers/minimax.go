// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// MiniMax-M2 uses a nested XML form:
//
//	<minimax:tool_call><invoke name="fn"><parameter name="k">v</parameter></invoke></minimax:tool_call>
//
// The partial patterns recover truncated streams where closing tags are missing.
var (
	minimaxBlockRE      = regexp.MustCompile(`(?s)<minimax:tool_call>(.*?)</minimax:tool_call>`)
	minimaxInvokeRE     = regexp.MustCompile(`(?s)<invoke\s+name="([^"]+)">(.*?)</invoke>`)
	minimaxInvokePartRE = regexp.MustCompile(`(?s)<invoke\s+name="([^"]+)">(.*)`)
	minimaxParamRE      = regexp.MustCompile(`(?s)<parameter\s+name="([^"]+)">(.*?)</parameter>`)
	minimaxParamPartRE  = regexp.MustCompile(`(?s)<parameter\s+name="([^"]+)">([^<]*)`)
	minimaxThinkRE      = regexp.MustCompile(`(?s)<think>.*?</think>`)
	minimaxArtifactRE   = regexp.MustCompile(`\[e~\[.*$`)
)

// MiniMaxParser handles the MiniMax-M2 native XML tool-call format.
type MiniMaxParser struct{ base }

// extractInvokes pulls calls out of an invoke region, tolerating truncation.
func (p *MiniMaxParser) extractInvokes(text string) []ToolCall {
	invokes := minimaxInvokeRE.FindAllStringSubmatch(text, -1)
	if len(invokes) == 0 {
		invokes = minimaxInvokePartRE.FindAllStringSubmatch(text, -1)
	}

	var calls []ToolCall
	for _, inv := range invokes {
		funcName, block := inv[1], inv[2]
		params := minimaxParamRE.FindAllStringSubmatch(block, -1)
		if len(params) == 0 {
			params = minimaxParamPartRE.FindAllStringSubmatch(block, -1)
		}
		if len(params) == 0 {
			continue // bare <invoke> with no parameters is junk
		}
		args := newOmap()
		for _, pm := range params {
			val := strip(pm[2])
			if val == "" {
				continue
			}
			if decoded, err := decodeOrdered(val); err == nil {
				args.set(pm[1], decoded)
			} else {
				args.set(pm[1], val)
			}
		}
		if args.len() == 0 {
			continue
		}
		calls = append(calls, ToolCall{ID: genToolID(), Name: strip(funcName), Arguments: dumps(args, false)})
	}
	return calls
}

func (p *MiniMaxParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	// Wrapped form first.
	blocks := minimaxBlockRE.FindAllStringSubmatch(modelOutput, -1)
	if len(blocks) > 0 {
		var calls []ToolCall
		for _, b := range blocks {
			calls = append(calls, p.extractInvokes(b[1])...)
		}
		cleaned := strip(minimaxBlockRE.ReplaceAllString(modelOutput, ""))
		cleaned = strip(minimaxThinkRE.ReplaceAllString(cleaned, ""))
		cleaned = strip(minimaxArtifactRE.ReplaceAllString(cleaned, ""))
		return ExtractedToolCalls{ToolsCalled: len(calls) > 0, ToolCalls: calls, Content: strNil(cleaned)}
	}

	// Bare invoke without the wrapper.
	if calls := p.extractInvokes(modelOutput); len(calls) > 0 {
		cleaned := strip(minimaxInvokeRE.ReplaceAllString(modelOutput, ""))
		cleaned = strip(minimaxThinkRE.ReplaceAllString(cleaned, ""))
		cleaned = strip(minimaxArtifactRE.ReplaceAllString(cleaned, ""))
		cleaned = strip(strings.ReplaceAll(cleaned, "</invoke>", ""))
		return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(cleaned)}
	}

	// Text-format fallback.
	if textCalls := extractTextFormatToolCalls(modelOutput); len(textCalls) > 0 {
		cleaned := textKVPattern.ReplaceAllString(modelOutput, "")
		cleaned = strip(textFnPattern.ReplaceAllString(cleaned, ""))
		cleaned = strip(minimaxThinkRE.ReplaceAllString(cleaned, ""))
		return ExtractedToolCalls{ToolsCalled: true, ToolCalls: textCalls, Content: strNil(cleaned)}
	}

	return ExtractedToolCalls{Content: ptr(modelOutput)}
}

func (p *MiniMaxParser) hasToolStart(text string) bool {
	return strings.Contains(text, "<minimax:tool_call>") ||
		(strings.Contains(text, `<invoke name="`) && minimaxInvokeRE.MatchString(text)) ||
		hasTextFormatToolCall(text)
}

func (p *MiniMaxParser) textTCCount(text string) int {
	return len(textKVPattern.FindAllString(text, -1)) + len(textFnPattern.FindAllString(text, -1))
}

func (p *MiniMaxParser) hasToolEnd(current, previous string) bool {
	if strings.Contains(current, "<minimax:tool_call>") {
		return strings.Count(current, "</minimax:tool_call>") > strings.Count(previous, "</minimax:tool_call>")
	}
	if strings.Count(current, "</invoke>") > strings.Count(previous, "</invoke>") {
		return true
	}
	return p.textTCCount(current) > p.textTCCount(previous)
}

func (p *MiniMaxParser) ExtractToolCallsStreaming(previousText, currentText, deltaText string, _ Request) *StreamDelta {
	if !p.hasToolStart(currentText) {
		return &StreamDelta{Content: ptr(deltaText)}
	}

	if p.hasToolEnd(currentText, previousText) {
		result := p.ExtractToolCalls(currentText, nil)
		if result.ToolsCalled {
			prevComplete := strings.Count(previousText, "</minimax:tool_call>")
			if prevComplete == 0 && !strings.Contains(previousText, "<minimax:tool_call>") {
				prevComplete = strings.Count(previousText, "</invoke>")
			}
			prevComplete += p.textTCCount(previousText)
			if len(result.ToolCalls) > prevComplete {
				return streamFromCalls(result.ToolCalls[prevComplete:], prevComplete)
			}
		}
	}
	return nil
}

func (p *MiniMaxParser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, "<minimax:tool_call>") ||
		strings.Contains(text, "<invoke name=") ||
		hasTextFormatToolCall(text)
}
