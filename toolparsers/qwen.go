// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// Qwen uses two shapes: an XML wrapper holding a JSON object, and a bracket
// "[Calling tool: name({json})]" text form.
var (
	qwenXMLRE     = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)
	qwenBracketRE = regexp.MustCompile(`(?s)\[Calling tool:\s*(\w+)\((\{.*?\})\)\]`)
)

// QwenParser handles the Qwen family JSON-wrapper and bracket text forms.
type QwenParser struct{ base }

func (p *QwenParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	cleaned := stripThinkTags(modelOutput)

	var calls []ToolCall

	bracket := qwenBracketRE.FindAllStringSubmatch(cleaned, -1)
	for _, m := range bracket {
		decoded, err := decodeOrdered(m[2])
		if err != nil {
			continue
		}
		calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: argString(decoded, false)})
	}
	if len(bracket) > 0 {
		cleaned = strip(qwenBracketRE.ReplaceAllString(cleaned, ""))
	}

	xml := qwenXMLRE.FindAllStringSubmatch(cleaned, -1)
	for _, m := range xml {
		decoded, err := decodeOrdered(m[1])
		if err != nil {
			continue
		}
		obj, ok := decoded.(*omap)
		if !ok {
			continue
		}
		name, _ := obj.get("name").(string)
		if name == "" {
			continue
		}
		args := obj.get("arguments")
		if args == nil {
			args = newOmap()
		}
		calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(args, false)})
	}
	if len(xml) > 0 {
		cleaned = strip(qwenXMLRE.ReplaceAllString(cleaned, ""))
	}

	if len(calls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(cleaned)}
}

func (p *QwenParser) ExtractToolCallsStreaming(previousText, currentText, deltaText string, req Request) *StreamDelta {
	if !strings.Contains(currentText, "<tool_call>") && !strings.Contains(currentText, "[Calling tool:") {
		return &StreamDelta{Content: ptr(deltaText)}
	}

	// Dedup against calls already emitted by counting closing markers. Re-emitting
	// every call on each closing marker would make the OpenAI stream merger glue
	// names and arguments together by index.
	prevClose := strings.Count(previousText, "</tool_call>") + strings.Count(previousText, ")]")
	curClose := strings.Count(currentText, "</tool_call>") + strings.Count(currentText, ")]")
	if curClose <= prevClose {
		return nil
	}

	result := p.ExtractToolCalls(currentText, req)
	if !result.ToolsCalled {
		return nil
	}
	prevResult := p.ExtractToolCalls(previousText, req)
	prevEmitted := 0
	if prevResult.ToolsCalled {
		prevEmitted = len(prevResult.ToolCalls)
	}
	if prevEmitted >= len(result.ToolCalls) {
		return nil
	}
	return streamFromCalls(result.ToolCalls[prevEmitted:], prevEmitted)
}
