// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// GLM-4.7 writes a function name on the first line of a <tool_call> block,
// then arg_key/arg_value tag pairs.
var (
	glmFuncDetailRE = regexp.MustCompile(`(?s)<tool_call>\s*([^\n<]+?)(?:\n|\s*)(<arg_key>.*?)?</tool_call>`)
	glmArgRE        = regexp.MustCompile(`(?s)<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>`)
)

// Glm47Parser handles GLM-4.7 named tool calls.
type Glm47Parser struct{ base }

// deserialize coerces an arg value via JSON, falling back to the raw string.
func glmDeserialize(value string) any {
	value = strip(value)
	if decoded, err := decodeOrdered(value); err == nil {
		return decoded
	}
	return value
}

// toolNames returns the set of valid function names declared in the request.
func toolNames(req Request) map[string]bool {
	names := map[string]bool{}
	for _, t := range req.tools() {
		fn, ok := t["function"].(map[string]any)
		if !ok {
			continue
		}
		name, _ := fn["name"].(string)
		names[name] = true
	}
	return names
}

func (p *Glm47Parser) ExtractToolCalls(modelOutput string, req Request) ExtractedToolCalls {
	cleaned := stripThinkTags(modelOutput)
	valid := toolNames(req)

	var calls []ToolCall
	for _, m := range glmFuncDetailRE.FindAllStringSubmatch(cleaned, -1) {
		funcName := strip(m[1])
		argsSection := m[2]
		if funcName == "" {
			continue
		}
		if len(valid) > 0 && !valid[funcName] {
			continue
		}

		args := newOmap()
		if argsSection != "" {
			for _, am := range glmArgRE.FindAllStringSubmatch(argsSection, -1) {
				key := strip(am[1])
				if key == "" {
					continue
				}
				args.set(key, glmDeserialize(am[2]))
			}
		}
		calls = append(calls, ToolCall{ID: genToolID(), Name: funcName, Arguments: dumps(args, false)})
	}

	if len(calls) == 0 {
		// GLM often emits reasoning before the tags; surface it as content.
		return ExtractedToolCalls{Content: ptr(cleaned)}
	}
	// When tool calls are found, the leading reasoning is dropped, not leaked.
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: nil}
}

func (p *Glm47Parser) ExtractToolCallsStreaming(_, currentText, deltaText string, req Request) *StreamDelta {
	// Suppress while inside an open think block.
	if strings.Contains(currentText, "<think>") && !strings.Contains(currentText, "</think>") {
		return nil
	}

	// Buffer until the tool-call block closes; do not leak reasoning as content.
	if strings.Contains(currentText, "<tool_call>") {
		if strings.Contains(deltaText, "</tool_call>") {
			return emitAllStreaming(p.ExtractToolCalls(currentText, req))
		}
		return nil
	}

	if strings.Contains(deltaText, "</think>") {
		clean := stripThinkTags(deltaText)
		if clean != "" {
			return &StreamDelta{Content: ptr(clean)}
		}
		return nil
	}
	if deltaText != "" {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	return nil
}
