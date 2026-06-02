// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// Llama emits each call as <function=name>{"arg": "value"}</function>.
var llamaFunctionRE = regexp.MustCompile(`(?s)<function=([^>]+)>(\{.*?\})</function>`)

// LlamaParser handles the Llama family bare-function wire format.
type LlamaParser struct{ base }

func (p *LlamaParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	matches := llamaFunctionRE.FindAllStringSubmatch(modelOutput, -1)
	if len(matches) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	var calls []ToolCall
	for _, m := range matches {
		name, argsStr := strip(m[1]), m[2]
		if decoded, err := decodeOrdered(argsStr); err == nil {
			calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(decoded, false)})
		} else {
			// Keep the raw argument string when it is not valid JSON.
			calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argsStr})
		}
	}

	cleaned := strip(llamaFunctionRE.ReplaceAllString(modelOutput, ""))
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(cleaned)}
}

func (p *LlamaParser) ExtractToolCallsStreaming(_, currentText, deltaText string, req Request) *StreamDelta {
	if !strings.Contains(currentText, "<function=") {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	if strings.Contains(deltaText, "</function>") {
		return emitAllStreaming(p.ExtractToolCalls(currentText, req))
	}
	return nil
}

func (p *LlamaParser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, "<function=")
}
