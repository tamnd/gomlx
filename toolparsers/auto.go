// SPDX-License-Identifier: Apache-2.0

package toolparsers

import "strings"

// AutoParser tries every known wire format in turn, so it works without
// knowing the model family ahead of time. The order matters: Mistral first
// (its marker is unambiguous), then the bracket text form, Nemotron XML before
// the more permissive Qwen/Hermes XML, Llama, and finally a raw-JSON fallback.
type AutoParser struct{ base }

// nameOrType returns the call name, falling back to a "type" field (Granite).
func nameOrType(o *omap) string {
	if n, _ := o.get("name").(string); n != "" {
		return n
	}
	if t, _ := o.get("type").(string); t != "" {
		return t
	}
	return ""
}

// argsOrEmpty returns the arguments value, or an empty object when absent.
func argsOrEmpty(o *omap) any {
	if o.has("arguments") {
		return o.get("arguments")
	}
	return newOmap()
}

func (p *AutoParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	var calls []ToolCall
	cleaned := modelOutput

	// 1. Mistral.
	if strings.Contains(modelOutput, mistralBotToken) {
		parts := strings.Split(modelOutput, mistralBotToken)
		content := strip(parts[0])
		for _, raw := range parts[1:] {
			raw = strip(raw)
			if raw == "" {
				continue
			}
			if !strings.HasPrefix(raw, "[") && strings.Contains(raw, "{") {
				endName := strings.Index(raw, "{")
				name := strip(raw[:endName])
				if name != "" {
					calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: raw[endName:]})
				}
				continue
			}
			if decoded, err := decodeOrdered(raw); err == nil {
				if arr, ok := decoded.([]any); ok {
					for _, item := range arr {
						if obj, ok := item.(*omap); ok && obj.has("name") {
							name, _ := obj.get("name").(string)
							calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(argsOrEmpty(obj), false)})
						}
					}
				}
			}
		}
		if len(calls) > 0 {
			return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(content)}
		}
	}

	// 2. Qwen bracket text form (scanned against the original output).
	bracket := qwenBracketRE.FindAllStringSubmatch(modelOutput, -1)
	for _, m := range bracket {
		if decoded, err := decodeOrdered(m[2]); err == nil {
			calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: argString(decoded, false)})
		} else {
			calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: m[2]})
		}
	}
	if len(bracket) > 0 {
		cleaned = strip(qwenBracketRE.ReplaceAllString(cleaned, ""))
	}

	// 3. Nemotron XML (before Qwen XML; it is the more specific shape).
	nem := nemotronCallRE.FindAllStringSubmatch(cleaned, -1)
	for _, m := range nem {
		args := newOmap()
		for _, pm := range nemotronParamRE.FindAllStringSubmatch(m[2], -1) {
			args.set(strip(pm[1]), strip(pm[2]))
		}
		calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: dumps(args, false)})
	}
	if len(nem) > 0 {
		cleaned = strip(nemotronCallRE.ReplaceAllString(cleaned, ""))
	}

	// 4. Qwen/Hermes XML with a JSON body.
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
		calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(argsOrEmpty(obj), false)})
	}
	if len(xml) > 0 {
		cleaned = strip(qwenXMLRE.ReplaceAllString(cleaned, ""))
	}

	// 5. Llama <function=name>{json}</function>.
	llama := llamaFunctionRE.FindAllStringSubmatch(cleaned, -1)
	for _, m := range llama {
		if decoded, err := decodeOrdered(m[2]); err == nil {
			calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: argString(decoded, false)})
		} else {
			calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: m[2]})
		}
	}
	if len(llama) > 0 {
		cleaned = strip(llamaFunctionRE.ReplaceAllString(cleaned, ""))
	}

	// 6. Raw JSON fallback.
	if len(calls) == 0 {
		if raw := parseRawJSONToolCalls(cleaned); len(raw) > 0 {
			calls = append(calls, raw...)
			cleaned = ""
		}
	}

	if len(calls) > 0 {
		return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(cleaned)}
	}
	return ExtractedToolCalls{Content: ptr(modelOutput)}
}

// parseRawJSONToolCalls recovers calls from a bare JSON array or from any
// balanced top-level objects embedded in text.
func parseRawJSONToolCalls(text string) []ToolCall {
	if text == "" {
		return nil
	}
	text = strip(text)
	var calls []ToolCall

	if strings.HasPrefix(text, "[") {
		if decoded, err := decodeOrdered(text); err == nil {
			if arr, ok := decoded.([]any); ok {
				for _, item := range arr {
					if obj, ok := item.(*omap); ok {
						if name := nameOrType(obj); name != "" {
							calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(argsOrEmpty(obj), false)})
						}
					}
				}
				return calls
			}
		}
	}

	depth := 0
	start := -1
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			depth--
			if depth < 0 {
				depth = 0
				start = -1
				continue
			}
			if depth == 0 && start >= 0 {
				if decoded, err := decodeOrdered(text[start : i+1]); err == nil {
					if obj, ok := decoded.(*omap); ok {
						if name := nameOrType(obj); name != "" {
							calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(argsOrEmpty(obj), false)})
						}
					}
				}
				start = -1
			}
		}
	}
	return calls
}

func (p *AutoParser) ExtractToolCallsStreaming(_, currentText, deltaText string, _ Request) *StreamDelta {
	markers := []string{mistralBotToken, "[Calling tool:", "<tool_call>", "<function="}
	hasMarker := false
	for _, m := range markers {
		if strings.Contains(currentText, m) {
			hasMarker = true
			break
		}
	}
	if !hasMarker {
		return &StreamDelta{Content: ptr(deltaText)}
	}

	for _, m := range []string{"</tool_call>", "</function>", ")]"} {
		if strings.Contains(deltaText, m) {
			if result := p.ExtractToolCalls(currentText, nil); result.ToolsCalled {
				return streamFromCalls(result.ToolCalls, 0)
			}
			break
		}
	}
	return nil
}

func (p *AutoParser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, mistralBotToken) ||
		strings.Contains(text, "[Calling tool:") ||
		strings.Contains(text, "<tool_call>") ||
		strings.Contains(text, "<function=") ||
		hasTextFormatToolCall(text)
}
