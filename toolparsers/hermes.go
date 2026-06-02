// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// Hermes is the most permissive parser: a JSON body inside <tool_call>, the
// Nemotron-style XML body, a bare <function=> block, and raw JSON, with a
// text-format fallback for low-quant degradation.
var (
	hermesLenientRE   = regexp.MustCompile(`(?s)<tool_call[^{]*(\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\})`)
	hermesReasoningRE = regexp.MustCompile(`(?s)<tool_call_reasoning>(.*?)</tool_call_reasoning>`)
	hermesRawJSONRE   = regexp.MustCompile(`(?s)\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}`)
	hermesBareFuncRE  = regexp.MustCompile(`(?s)<function=([^>]+)>(.*?)</function>`)
)

// HermesParser handles the Hermes/Nous family and its many fallbacks.
type HermesParser struct{ base }

// parseParamValue coerces an XML parameter value, preferring JSON, then a few
// Python literal forms, then the raw string.
func parseParamValue(val string) any {
	if decoded, err := decodeOrdered(val); err == nil {
		return decoded
	}
	switch strip(val) {
	case "True":
		return true
	case "False":
		return false
	case "None":
		return nil
	}
	if t := strip(val); len(t) >= 2 && t[0] == '\'' && t[len(t)-1] == '\'' {
		return t[1 : len(t)-1]
	}
	return val
}

// xmlParams turns a <parameter=...> block into an ordered object.
func xmlParams(block string) *omap {
	args := newOmap()
	for _, pm := range nemotronParamRE.FindAllStringSubmatch(block, -1) {
		args.set(strip(pm[1]), parseParamValue(strip(pm[2])))
	}
	return args
}

func (p *HermesParser) ExtractToolCalls(modelOutput string, req Request) ExtractedToolCalls {
	cleaned := stripThinkTags(modelOutput)

	// Pull reasoning aside; it is folded back into content at the end.
	reasoning := hermesReasoningRE.FindAllStringSubmatch(cleaned, -1)
	cleaned = hermesReasoningRE.ReplaceAllString(cleaned, "")

	var calls []ToolCall

	// Primary: JSON body inside <tool_call>.
	jsonMatches := qwenXMLRE.FindAllStringSubmatch(cleaned, -1)
	for _, m := range jsonMatches {
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
	if len(jsonMatches) > 0 {
		cleaned = strip(qwenXMLRE.ReplaceAllString(cleaned, ""))
	}

	// Nemotron XML body.
	if len(calls) == 0 {
		matches := nemotronCallRE.FindAllStringSubmatch(cleaned, -1)
		for _, m := range matches {
			calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: dumps(xmlParams(m[2]), false)})
		}
		if len(matches) > 0 {
			cleaned = strip(nemotronCallRE.ReplaceAllString(cleaned, ""))
		}
	}

	// Bare <function=> block.
	if len(calls) == 0 {
		matches := hermesBareFuncRE.FindAllStringSubmatch(cleaned, -1)
		for _, m := range matches {
			calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: dumps(xmlParams(m[2]), false)})
		}
		if len(matches) > 0 {
			cleaned = strip(hermesBareFuncRE.ReplaceAllString(cleaned, ""))
		}
	}

	// Lenient: malformed <tool_call without a >. Only the first, to avoid
	// hallucinated extras.
	if len(calls) == 0 {
		matches := hermesLenientRE.FindAllStringSubmatch(cleaned, -1)
		if len(matches) > 0 {
			if decoded, err := decodeOrdered(matches[0][1]); err == nil {
				if obj, ok := decoded.(*omap); ok {
					if name, _ := obj.get("name").(string); name != "" {
						args := obj.get("arguments")
						if args == nil {
							args = newOmap()
						}
						calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(args, false)})
						cleaned = strip(replaceFirst(hermesLenientRE, cleaned))
					}
				}
			}
		}
	}

	// Raw JSON, first valid call only.
	if len(calls) == 0 {
		matches := hermesRawJSONRE.FindAllStringSubmatch(cleaned, -1)
		if len(matches) > 0 {
			name, argsStr := matches[0][1], matches[0][2]
			if decoded, err := decodeOrdered(argsStr); err == nil {
				valid := true
				if names := toolNames(req); len(names) > 0 {
					valid = names[name]
				}
				if valid && name != "" {
					calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(decoded, false)})
					cleaned = strip(replaceFirst(hermesRawJSONRE, cleaned))
				}
			}
		}
	}

	// Fold reasoning back into the content.
	if len(reasoning) > 0 {
		texts := make([]string, len(reasoning))
		for i, m := range reasoning {
			texts[i] = m[1]
		}
		joined := strings.Join(texts, " ")
		if cleaned != "" {
			cleaned = cleaned + "\n\n(Reasoning: " + joined + ")"
		} else {
			cleaned = "(Reasoning: " + joined + ")"
		}
	}

	if len(calls) == 0 {
		return ExtractedToolCalls{Content: ptr(cleaned)}
	}
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(cleaned)}
}

func (p *HermesParser) ExtractToolCallsStreaming(previousText, currentText, deltaText string, req Request) *StreamDelta {
	openCount := strings.Count(currentText, "<tool_call>")
	closeCount := strings.Count(currentText, "</tool_call>")
	prevClose := strings.Count(previousText, "</tool_call>")

	if openCount > 0 {
		if openCount > closeCount {
			return nil
		}
		if closeCount > prevClose {
			result := p.ExtractToolCalls(currentText, req)
			if result.ToolsCalled && len(result.ToolCalls) > prevClose {
				return streamFromCalls(result.ToolCalls[prevClose:], prevClose)
			}
		}
		return &StreamDelta{Content: ptr(deltaText)}
	}

	if strings.Contains(currentText, "<function=") {
		funcClose := strings.Count(currentText, "</function>")
		prevFuncClose := strings.Count(previousText, "</function>")
		if strings.Count(currentText, "<function=") > funcClose {
			return nil
		}
		if funcClose > prevFuncClose {
			result := p.ExtractToolCalls(currentText, req)
			if result.ToolsCalled && len(result.ToolCalls) > prevFuncClose {
				return streamFromCalls(result.ToolCalls[prevFuncClose:], prevFuncClose)
			}
		}
		return &StreamDelta{Content: ptr(deltaText)}
	}

	if strings.Contains(currentText, `{"name":`) && strings.Contains(currentText, `"arguments":`) {
		if strings.HasSuffix(strings.TrimRight(deltaText, " \t\n\r\f\v"), "}") {
			result := p.ExtractToolCalls(currentText, req)
			if result.ToolsCalled {
				return streamFromCalls(result.ToolCalls, 0)
			}
		}
		return nil
	}

	return &StreamDelta{Content: ptr(deltaText)}
}
