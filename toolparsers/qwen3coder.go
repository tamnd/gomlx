// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strconv"
	"strings"
)

// Qwen3-Coder uses the same XML body as Seed-OSS but without the seed:
// namespace. Parameter values are typed from the tool schema, and bare JSON-
// looking values are decoded one or two levels deep.
const (
	q3cStart       = "<tool_call>"
	q3cEnd         = "</tool_call>"
	q3cPrefix      = "<function="
	q3cFuncEnd     = "</function>"
	q3cParamPrefix = "<parameter="
	q3cParamEnd    = "</parameter>"
)

var q3cToolCallRE = regexp.MustCompile(`(?s)<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$`)

// decodeJSONLike decodes a JSON-looking string, unwrapping up to one extra
// level of double encoding. It mirrors api.tool_calling._decode_json_like.
func decodeJSONLike(value string) any {
	var current any = strings.TrimSpace(value)
	for i := 0; i < 3; i++ {
		s, ok := current.(string)
		if !ok {
			return current
		}
		stripped := strings.TrimSpace(s)
		if stripped == "" || !strings.ContainsRune(`[{"`, rune(stripped[0])) {
			return current
		}
		parsed, err := decodeOrdered(stripped)
		if err != nil {
			return current
		}
		if ps, ok := parsed.(string); ok && ps == s {
			return parsed
		}
		current = parsed
	}
	return current
}

// schemaType resolves the effective JSON-schema type of a parameter, following
// type lists and anyOf/oneOf/allOf. It mirrors api.tool_calling._schema_type.
func schemaType(schema any) (string, bool) {
	if s, ok := schema.(string); ok {
		return strings.ToLower(strip(s)), true
	}
	m, ok := schema.(map[string]any)
	if !ok {
		return "", false
	}
	st := m["type"]
	if arr, ok := st.([]any); ok {
		st = nil
		for _, item := range arr {
			if s, ok := item.(string); ok && s != "null" {
				st = s
				break
			}
		}
	}
	if s, ok := st.(string); ok {
		return strings.ToLower(strip(s)), true
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if opts, ok := m[key].([]any); ok {
			for _, option := range opts {
				if t, ok := schemaType(option); ok && t != "" && t != "null" {
					return t, true
				}
			}
		}
	}
	if _, ok := m["items"]; ok {
		return "array", true
	}
	if _, ok := m["properties"]; ok {
		return "object", true
	}
	if _, ok := m["additionalProperties"]; ok {
		return "object", true
	}
	if _, ok := m["enum"]; ok {
		return "string", true
	}
	return "", false
}

// qwenArgumentsConfig returns the property schema for func, or an empty map.
func qwenArgumentsConfig(funcName string, req Request) map[string]any {
	for _, tool := range req.tools() {
		fn, _ := tool["function"].(map[string]any)
		if fn == nil {
			continue
		}
		if name, _ := fn["name"].(string); name == funcName {
			if params, ok := fn["parameters"].(map[string]any); ok {
				if props, ok := params["properties"].(map[string]any); ok {
					return props
				}
			}
			return map[string]any{}
		}
	}
	return map[string]any{}
}

// qwenConvertParamValue coerces a raw XML parameter value using the tool
// schema, decoding JSON-like values when no concrete type applies.
func qwenConvertParamValue(pv, paramName string, config map[string]any) any {
	if strings.ToLower(pv) == "null" {
		return nil
	}
	cfg, ok := config[paramName]
	if !ok {
		return decodeJSONLike(pv)
	}
	paramType, ok := schemaType(cfg)
	if !ok {
		return decodeJSONLike(pv)
	}
	switch {
	case paramType == "string" || paramType == "str" || paramType == "text" ||
		paramType == "varchar" || paramType == "char" || paramType == "enum":
		return pv
	case hasAnyPrefix(paramType, "int", "uint", "long", "short", "unsigned"):
		if n, err := strconv.Atoi(strings.TrimSpace(pv)); err == nil {
			return n
		}
		return pv
	case hasAnyPrefix(paramType, "num", "float", "double"):
		if f, err := strconv.ParseFloat(strings.TrimSpace(pv), 64); err == nil {
			return f
		}
		return pv
	case paramType == "boolean" || paramType == "bool" || paramType == "binary":
		return strings.ToLower(pv) == "true"
	default:
		if paramType == "object" || paramType == "array" || paramType == "arr" ||
			hasAnyPrefix(paramType, "dict", "list") {
			decoded := decodeJSONLike(pv)
			if ds, ok := decoded.(string); !ok || ds != pv {
				return decoded
			}
		}
		if v, ok := pyLiteralEval(pv); ok {
			return v
		}
		return pv
	}
}

// qwenFindParams scans a function body for <parameter=...> entries, returning
// each "NAME>VALUE" fragment. The reference regex relies on lookahead, which
// RE2 lacks, so the boundaries are found by hand: a parameter ends at the
// nearest </parameter>, next <parameter=, </function>, or end of text.
func qwenFindParams(parameters string) []string {
	var out []string
	pos := 0
	for {
		s := strings.Index(parameters[pos:], q3cParamPrefix)
		if s < 0 {
			break
		}
		start := pos + s + len(q3cParamPrefix)
		rest := parameters[start:]
		end := len(rest)
		for _, term := range []string{q3cParamEnd, q3cParamPrefix, q3cFuncEnd} {
			if i := strings.Index(rest, term); i >= 0 && i < end {
				end = i
			}
		}
		out = append(out, rest[:end])
		pos = start + end
		if strings.HasPrefix(parameters[pos:], q3cParamEnd) {
			pos += len(q3cParamEnd)
		}
	}
	return out
}

// Qwen3CoderParser handles the Qwen3-Coder XML tool-call format.
type Qwen3CoderParser struct {
	base
	currentToolIndex    int
	isToolCallStarted   bool
	headerSent          bool
	currentGenID        string
	currentFunctionName string
	paramCount          int
	inParam             bool
	inFunction          bool
	accumulatedText     string
	jsonStarted         bool
	jsonClosed          bool
	accumulatedParams   map[string]string
	streamingRequest    Request
}

func (p *Qwen3CoderParser) Reset() {
	p.base.Reset()
	p.currentToolIndex = 0
	p.isToolCallStarted = false
	p.headerSent = false
	p.currentGenID = ""
	p.currentFunctionName = ""
	p.paramCount = 0
	p.inParam = false
	p.inFunction = false
	p.accumulatedText = ""
	p.jsonStarted = false
	p.jsonClosed = false
	p.accumulatedParams = map[string]string{}
	p.streamingRequest = nil
	p.prevToolCallArr = nil
}

func (p *Qwen3CoderParser) parseXMLFunctionCall(fcStr string, req Request) *ToolCall {
	endIndex := strings.Index(fcStr, ">")
	if endIndex < 0 {
		return nil
	}
	functionName := fcStr[:endIndex]
	config := qwenArgumentsConfig(functionName, req)
	parameters := fcStr[endIndex+1:]
	paramDict := newOmap()
	for _, matchText := range qwenFindParams(parameters) {
		idx := strings.Index(matchText, ">")
		if idx < 0 {
			continue
		}
		pName := matchText[:idx]
		pValue := matchText[idx+1:]
		pValue = strings.TrimPrefix(pValue, "\n")
		pValue = strings.TrimSuffix(pValue, "\n")
		paramDict.set(pName, qwenConvertParamValue(pValue, pName, config))
	}
	return &ToolCall{ID: genToolID(), Name: functionName, Arguments: dumps(paramDict, false)}
}

func (p *Qwen3CoderParser) functionCalls(modelOutput string) []string {
	var raw []string
	for _, m := range q3cToolCallRE.FindAllStringSubmatch(modelOutput, -1) {
		if m[1] != "" {
			raw = append(raw, m[1])
		} else {
			raw = append(raw, m[2])
		}
	}
	if len(raw) == 0 {
		raw = []string{modelOutput}
	}
	var fns []string
	for _, tc := range raw {
		for _, m := range seedFunctionRE.FindAllStringSubmatch(tc, -1) {
			if m[1] != "" {
				fns = append(fns, m[1])
			} else {
				fns = append(fns, m[2])
			}
		}
	}
	return fns
}

func (p *Qwen3CoderParser) ExtractToolCalls(modelOutput string, req Request) ExtractedToolCalls {
	if !strings.Contains(modelOutput, q3cPrefix) {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	functionCalls := p.functionCalls(modelOutput)
	if len(functionCalls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	var calls []ToolCall
	for _, fc := range functionCalls {
		if tc := p.parseXMLFunctionCall(fc, req); tc != nil {
			calls = append(calls, *tc)
		}
	}

	contentIndex := strings.Index(modelOutput, q3cStart)
	if contentIndex < 0 {
		contentIndex = strings.Index(modelOutput, q3cPrefix)
	}
	content := modelOutput[:contentIndex]

	return ExtractedToolCalls{
		ToolsCalled: len(calls) > 0,
		ToolCalls:   calls,
		Content:     strNil(content),
	}
}

func (p *Qwen3CoderParser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, q3cStart) ||
		strings.Contains(text, q3cPrefix) ||
		hasTextFormatToolCall(text)
}

func (p *Qwen3CoderParser) ExtractToolCallsStreaming(previousText, currentText, deltaText string, req Request) *StreamDelta {
	if previousText == "" {
		p.Reset()
		p.streamingRequest = req
	} else if req != nil && p.streamingRequest == nil {
		p.streamingRequest = req
	}
	if deltaText == "" {
		return nil
	}
	p.accumulatedText = currentText

	// Advance to the next tool once the current one has closed.
	if p.jsonClosed && !p.inFunction {
		toolEnds := strings.Count(currentText, q3cEnd)
		if toolEnds > p.currentToolIndex {
			p.currentToolIndex++
			p.headerSent = false
			p.paramCount = 0
			p.jsonStarted = false
			p.jsonClosed = false
			p.accumulatedParams = map[string]string{}
			if p.currentToolIndex >= strings.Count(currentText, q3cStart) {
				p.isToolCallStarted = false
			}
			return nil
		}
	}

	// Stream plain content until a tool call starts.
	if !p.isToolCallStarted {
		if strings.Contains(deltaText, q3cStart) {
			p.isToolCallStarted = true
			contentBefore := deltaText[:strings.Index(deltaText, q3cStart)]
			if contentBefore != "" {
				return &StreamDelta{Content: ptr(contentBefore)}
			}
			// Fall through; the header may already be in current_text.
		} else {
			if strings.HasSuffix(strings.TrimRight(currentText, " \t\n\r\f\v"), q3cEnd) && strip(deltaText) == "" {
				return nil
			}
			return &StreamDelta{Content: ptr(deltaText)}
		}
	}

	toolStartsCount := strings.Count(currentText, q3cStart)
	if p.currentToolIndex >= toolStartsCount {
		return nil
	}

	var toolStarts []int
	idx := 0
	for {
		j := strings.Index(currentText[idx:], q3cStart)
		if j < 0 {
			break
		}
		abs := idx + j
		toolStarts = append(toolStarts, abs)
		idx = abs + len(q3cStart)
	}
	if p.currentToolIndex >= len(toolStarts) {
		return nil
	}
	toolStartIdx := toolStarts[p.currentToolIndex]
	var toolText string
	if te := strings.Index(currentText[toolStartIdx:], q3cEnd); te < 0 {
		toolText = currentText[toolStartIdx:]
	} else {
		toolText = currentText[toolStartIdx : toolStartIdx+te+len(q3cEnd)]
	}

	// Parse and emit the function header once.
	if !p.headerSent {
		if strings.Contains(toolText, q3cPrefix) {
			funcStart := strings.Index(toolText, q3cPrefix) + len(q3cPrefix)
			if funcEnd := strings.Index(toolText[funcStart:], ">"); funcEnd != -1 {
				p.currentFunctionName = toolText[funcStart : funcStart+funcEnd]
				p.currentGenID = genToolID()
				p.headerSent = true
				p.inFunction = true

				if strings.Contains(toolText, q3cFuncEnd) {
					fcEnd := strings.Index(toolText[funcStart:], q3cFuncEnd)
					fc := toolText[funcStart : funcStart+fcEnd]
					args := "{}"
					if parsed := p.parseXMLFunctionCall(fc, p.streamingRequest); parsed != nil {
						args = parsed.Arguments
					}
					p.jsonStarted = true
					p.jsonClosed = true
					p.inFunction = false
					p.accumulatedParams = map[string]string{}
					p.prevToolCallArr = append(p.prevToolCallArr, map[string]any{"name": p.currentFunctionName, "arguments": args})
					return &StreamDelta{ToolCalls: []StreamToolCall{{
						Index:    p.currentToolIndex,
						ID:       ptr(p.currentGenID),
						Type:     ptr(typeFunction),
						Function: &StreamFunction{Name: ptr(p.currentFunctionName), Arguments: ptr(args)},
					}}}
				}

				p.prevToolCallArr = append(p.prevToolCallArr, map[string]any{"name": p.currentFunctionName, "arguments": "{}"})
				return &StreamDelta{ToolCalls: []StreamToolCall{{
					Index:    p.currentToolIndex,
					ID:       ptr(p.currentGenID),
					Type:     ptr(typeFunction),
					Function: &StreamFunction{Name: ptr(p.currentFunctionName), Arguments: ptr("")},
				}}}
			}
		}
		return nil
	}

	// Stream the function body.
	if p.inFunction {
		if !p.jsonStarted {
			p.jsonStarted = true
			return &StreamDelta{ToolCalls: []StreamToolCall{{
				Index:    p.currentToolIndex,
				Function: &StreamFunction{Arguments: ptr("{")},
			}}}
		}

		// Find parameter start positions within the tool text.
		var paramStarts []int
		si := 0
		for {
			j := strings.Index(toolText[si:], q3cParamPrefix)
			if j < 0 {
				break
			}
			abs := si + j
			paramStarts = append(paramStarts, abs)
			si = abs + len(q3cParamPrefix)
		}

		// Emit every newly completed parameter as one combined fragment.
		var fragments []string
		for !p.inParam && p.paramCount < len(paramStarts) {
			paramStart := paramStarts[p.paramCount] + len(q3cParamPrefix)
			remaining := toolText[paramStart:]
			if !strings.Contains(remaining, ">") {
				break
			}
			nameEnd := strings.Index(remaining, ">")
			currentParamName := remaining[:nameEnd]
			valueText := toolText[paramStart+nameEnd+1:]
			valueText = strings.TrimPrefix(valueText, "\n")

			paramEndIdx := strings.Index(valueText, q3cParamEnd)
			if paramEndIdx == -1 {
				nextParam := strings.Index(valueText, q3cParamPrefix)
				funcEnd := strings.Index(valueText, q3cFuncEnd)
				switch {
				case nextParam != -1 && (funcEnd == -1 || nextParam < funcEnd):
					paramEndIdx = nextParam
				case funcEnd != -1:
					paramEndIdx = funcEnd
				default:
					if toolEnd := strings.Index(valueText, q3cEnd); toolEnd != -1 {
						paramEndIdx = toolEnd
					} else {
						break
					}
				}
			}
			if paramEndIdx == -1 {
				break
			}

			pv := strings.TrimSuffix(valueText[:paramEndIdx], "\n")
			p.accumulatedParams[currentParamName] = pv

			config := qwenArgumentsConfig(p.currentFunctionName, p.streamingRequest)
			serialized := dumps(qwenConvertParamValue(pv, currentParamName, config), false)
			frag := `"` + currentParamName + `": ` + serialized
			if p.paramCount != 0 {
				frag = ", " + frag
			}
			p.paramCount++
			fragments = append(fragments, frag)
		}

		if len(fragments) > 0 {
			return &StreamDelta{ToolCalls: []StreamToolCall{{
				Index:    p.currentToolIndex,
				Function: &StreamFunction{Arguments: ptr(strings.Join(fragments, ""))},
			}}}
		}

		// Close the JSON object once the function end arrives.
		if !p.jsonClosed && strings.Contains(toolText, q3cFuncEnd) {
			p.jsonClosed = true
			funcStart := strings.Index(toolText, q3cPrefix) + len(q3cPrefix)
			if fcEnd := strings.Index(toolText[funcStart:], q3cFuncEnd); fcEnd != -1 {
				fc := toolText[funcStart : funcStart+fcEnd]
				if parsed := p.parseXMLFunctionCall(fc, p.streamingRequest); parsed != nil && p.currentToolIndex < len(p.prevToolCallArr) {
					p.prevToolCallArr[p.currentToolIndex]["arguments"] = parsed.Arguments
				}
			}
			p.inFunction = false
			p.accumulatedParams = map[string]string{}
			return &StreamDelta{ToolCalls: []StreamToolCall{{
				Index:    p.currentToolIndex,
				Function: &StreamFunction{Arguments: ptr("}")},
			}}}
		}
	}

	return nil
}
