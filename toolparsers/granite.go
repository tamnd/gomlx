// SPDX-License-Identifier: Apache-2.0

package toolparsers

import "strings"

const (
	graniteBotToken  = "<|tool_call|>"
	graniteBotString = "<tool_call>"
)

// GraniteParser handles IBM Granite output: an optional marker followed by a
// JSON array of {"name", "arguments"} objects.
type GraniteParser struct{ base }

func (p *GraniteParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	none := ExtractedToolCalls{Content: ptr(modelOutput)}

	stripped := strip(modelOutput)
	switch {
	case strings.HasPrefix(stripped, graniteBotToken):
		stripped = strings.TrimLeft(stripped[len(graniteBotToken):], " \t\n\r\f\v")
	case strings.HasPrefix(stripped, graniteBotString):
		stripped = strings.TrimLeft(stripped[len(graniteBotString):], " \t\n\r\f\v")
	}

	if stripped == "" || stripped[0] != '[' {
		return none
	}

	decoded, err := decodeOrdered(stripped)
	if err != nil {
		return none
	}
	arr, ok := decoded.([]any)
	if !ok {
		return none
	}

	var calls []ToolCall
	for _, item := range arr {
		call, ok := item.(*omap)
		if !ok {
			continue
		}
		// Granite uses "name" or "type" for the function name.
		funcName, _ := call.get("name").(string)
		if funcName == "" {
			funcName, _ = call.get("type").(string)
		}
		if funcName == "" {
			continue
		}
		args := call.get("arguments")
		if args == nil {
			args = newOmap()
		}
		calls = append(calls, ToolCall{ID: genToolID(), Name: funcName, Arguments: argString(args, false)})
	}

	if len(calls) == 0 {
		return none
	}
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: nil}
}

func (p *GraniteParser) ExtractToolCallsStreaming(_, currentText, deltaText string, req Request) *StreamDelta {
	stripped := strip(currentText)
	if !strings.HasPrefix(stripped, graniteBotToken) && !strings.HasPrefix(stripped, graniteBotString) {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	if strings.Contains(deltaText, "]") {
		return emitAllStreaming(p.ExtractToolCalls(currentText, req))
	}
	return nil
}
