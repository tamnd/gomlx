// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// Nemotron wraps each call as
// <tool_call><function=name>BODY</function></tool_call>, where BODY is either a
// JSON object or a sequence of <parameter=key>value</parameter> tags.
var (
	nemotronCallRE  = regexp.MustCompile(`(?s)<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>`)
	nemotronParamRE = regexp.MustCompile(`(?s)<parameter=([^>]+)>\s*(.*?)\s*</parameter>`)
)

// NemotronParser handles NVIDIA Nemotron XML-body tool calls.
type NemotronParser struct{ base }

func (p *NemotronParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	if !strings.Contains(modelOutput, "<tool_call>") {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	matches := nemotronCallRE.FindAllStringSubmatch(modelOutput, -1)
	var calls []ToolCall
	for _, m := range matches {
		funcName := strip(m[1])
		content := strip(m[2])

		// A JSON object body is passed through as-is.
		if strings.HasPrefix(content, "{") {
			if _, err := decodeOrdered(content); err == nil {
				calls = append(calls, ToolCall{ID: genToolID(), Name: funcName, Arguments: content})
				continue
			}
		}

		// Otherwise collect <parameter=...> tags into an ordered object.
		params := nemotronParamRE.FindAllStringSubmatch(content, -1)
		if len(params) > 0 {
			args := newOmap()
			for _, pm := range params {
				key := strip(pm[1])
				val := strip(pm[2])
				if decoded, err := decodeOrdered(val); err == nil {
					args.set(key, decoded)
				} else {
					args.set(key, val)
				}
			}
			calls = append(calls, ToolCall{ID: genToolID(), Name: funcName, Arguments: dumps(args, false)})
		} else if content != "" {
			calls = append(calls, ToolCall{ID: genToolID(), Name: funcName, Arguments: content})
		}
	}

	if len(calls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}
	cleaned := strip(nemotronCallRE.ReplaceAllString(modelOutput, ""))
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(cleaned)}
}

func (p *NemotronParser) ExtractToolCallsStreaming(_, currentText, deltaText string, req Request) *StreamDelta {
	if !strings.Contains(currentText, "<tool_call>") {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	if strings.Contains(deltaText, "</tool_call>") {
		return emitAllStreaming(p.ExtractToolCalls(currentText, req))
	}
	return nil
}
