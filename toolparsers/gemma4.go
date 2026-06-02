// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strconv"
	"strings"
)

// Gemma 4 writes <|tool_call>call:name{key:<|"|>value<|"|>, num:3}<tool_call|>.
// String values are wrapped in quote tokens; numbers/bools/null are bare.
var (
	gemma4ToolRE   = regexp.MustCompile(`(?s)<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>`)
	gemma4QuotedRE = regexp.MustCompile(`(?s)<\|"\|>(.*?)<\|"\|>`)
	gemma4BareKVRE = regexp.MustCompile(`(\w+)\s*:\s*([^,]+?)\s*(?:,|$)`)
)

// Gemma4Parser handles the Gemma 4 native tool-call format.
type Gemma4Parser struct {
	base
	emittedToolCount int
}

func (p *Gemma4Parser) Reset() {
	p.base.Reset()
	p.emittedToolCount = 0
}

// parseGemma4Args parses Gemma 4's brace block into an ordered object. Quoted
// strings are stashed behind placeholders so the bare key/value scan cannot
// trip over their contents, then restored.
func parseGemma4Args(argsStr string) *omap {
	var stashed []string
	cleaned := gemma4QuotedRE.ReplaceAllStringFunc(argsStr, func(m string) string {
		sub := gemma4QuotedRE.FindStringSubmatch(m)
		stashed = append(stashed, sub[1])
		return "__Q" + strconv.Itoa(len(stashed)-1) + "__"
	})

	result := newOmap()
	for _, kv := range gemma4BareKVRE.FindAllStringSubmatch(cleaned, -1) {
		key := kv[1]
		rawVal := strip(kv[2])
		if strings.HasPrefix(rawVal, "__Q") && strings.HasSuffix(rawVal, "__") {
			if idx, err := strconv.Atoi(rawVal[3 : len(rawVal)-2]); err == nil && idx >= 0 && idx < len(stashed) {
				result.set(key, stashed[idx])
				continue
			}
		}
		if decoded, err := decodeOrdered(rawVal); err == nil {
			result.set(key, decoded)
		} else {
			result.set(key, rawVal)
		}
	}
	return result
}

func (p *Gemma4Parser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	matches := gemma4ToolRE.FindAllStringSubmatch(modelOutput, -1)
	if len(matches) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	var calls []ToolCall
	for _, m := range matches {
		args := parseGemma4Args(m[2])
		// Gemma 4 serializes with json.dumps defaults, so ensure_ascii is on.
		calls = append(calls, ToolCall{ID: genToolID(), Name: m[1], Arguments: dumps(args, true)})
	}

	content := strNil(strip(gemma4ToolRE.ReplaceAllString(modelOutput, "")))
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: content}
}

func (p *Gemma4Parser) ExtractToolCallsStreaming(_, currentText, deltaText string, _ Request) *StreamDelta {
	if strings.Contains(currentText, "<|tool_call>") {
		completed := strings.Count(currentText, "<tool_call|>")
		open := strings.Count(currentText, "<|tool_call>")

		if completed < open {
			return nil // still accumulating an open call
		}
		if completed <= p.emittedToolCount {
			return nil
		}
		result := p.ExtractToolCalls(currentText, nil)
		if result.ToolsCalled {
			newCalls := result.ToolCalls[p.emittedToolCount:]
			p.emittedToolCount = len(result.ToolCalls)
			if len(newCalls) > 0 {
				return streamFromCalls(newCalls, p.emittedToolCount-len(newCalls))
			}
		}
		// Fall through to the text-format recovery below if nothing emitted.
	}

	// Text-format recovery: models degrade to [Calling tool: name({...})].
	if textAny.MatchString(currentText) {
		matches := textFnPattern.FindAllStringSubmatch(currentText, -1)
		if len(matches) > p.emittedToolCount {
			newMatches := matches[p.emittedToolCount:]
			p.emittedToolCount = len(matches)
			start := p.emittedToolCount - len(newMatches)
			entries := make([]StreamToolCall, len(newMatches))
			for i, m := range newMatches {
				entries[i] = StreamToolCall{
					Index:    start + i,
					ID:       ptr(genToolID()),
					Type:     ptr(typeFunction),
					Function: &StreamFunction{Name: ptr(m[1]), Arguments: ptr(m[2])},
				}
			}
			return &StreamDelta{ToolCalls: entries}
		}
		return nil
	}

	return &StreamDelta{Content: ptr(deltaText)}
}

func (p *Gemma4Parser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, "<|tool_call>") || hasTextFormatToolCall(text)
}
