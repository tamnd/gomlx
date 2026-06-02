// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Seed-OSS wraps each call in <seed:tool_call> with an inner <function=NAME>
// header and <parameter=KEY>VALUE</parameter> entries. Parameter values are
// typed from the request's tool schema, so a value declared "integer" is
// emitted as a JSON number rather than a quoted string. Thinking lives in
// <seed:think>...</seed:think>.
const (
	seedStart       = "<seed:tool_call>"
	seedEnd         = "</seed:tool_call>"
	seedPrefix      = "<function="
	seedFuncEnd     = "</function>"
	seedParamPrefix = "<parameter="
	seedParamEnd    = "</parameter>"
	seedThinkStart  = "<seed:think>"
	seedThinkEnd    = "</seed:think>"
)

var (
	seedToolCallRE  = regexp.MustCompile(`(?s)<seed:tool_call>(.*?)</seed:tool_call>|<seed:tool_call>(.*?)$`)
	seedFunctionRE  = regexp.MustCompile(`(?s)<function=(.*?)</function>|<function=(.*)$`)
	seedParameterRE = regexp.MustCompile(`(?s)<parameter=(.*?)</parameter>|<parameter=(.*?)$`)
)

// SeedOssParser handles the Seed-OSS / GPT-OSS XML tool-call format.
type SeedOssParser struct {
	base
	currentToolIndex    int
	isToolCallStarted   bool
	isThinkingEnd       bool
	headerSent          bool
	currentGenID        string
	currentFunctionName string
	paramCount          int
	inParam             bool
	inFunction          bool
	accumulatedText     string
	jsonStarted         bool
	jsonClosed          bool
}

func (p *SeedOssParser) Reset() {
	p.base.Reset()
	p.currentToolIndex = 0
	p.isToolCallStarted = false
	p.isThinkingEnd = false
	p.headerSent = false
	p.currentGenID = ""
	p.currentFunctionName = ""
	p.paramCount = 0
	p.inParam = false
	p.inFunction = false
	p.accumulatedText = ""
	p.jsonStarted = false
	p.jsonClosed = false
	p.prevToolCallArr = nil
}

// seedArgumentsConfig pulls the property schema for func from the request tools,
// used for parameter type coercion.
func seedArgumentsConfig(funcName string, req Request) map[string]any {
	for _, tool := range req.tools() {
		fn, _ := tool["function"].(map[string]any)
		if fn == nil {
			continue
		}
		if name, _ := fn["name"].(string); name == funcName {
			params, ok := fn["parameters"].(map[string]any)
			if !ok {
				return map[string]any{}
			}
			if props, ok := params["properties"].(map[string]any); ok {
				return props
			}
			return params
		}
	}
	return map[string]any{}
}

func hasAnyPrefix(s string, prefixes ...string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

// pyLiteralEval approximates ast.literal_eval for the small set of forms a
// parameter value can take when no schema type applies.
func pyLiteralEval(s string) (any, bool) {
	t := strings.TrimSpace(s)
	switch t {
	case "True":
		return true, true
	case "False":
		return false, true
	case "None":
		return nil, true
	}
	if n, err := strconv.Atoi(t); err == nil {
		return n, true
	}
	if f, err := strconv.ParseFloat(t, 64); err == nil {
		return f, true
	}
	if len(t) >= 2 {
		if (t[0] == '\'' && t[len(t)-1] == '\'') || (t[0] == '"' && t[len(t)-1] == '"') {
			return t[1 : len(t)-1], true
		}
	}
	if d, err := decodeOrdered(t); err == nil {
		return d, true
	}
	return nil, false
}

// convertParamValue coerces a raw XML parameter value to the type declared in
// the tool schema, falling back to the raw string when no type matches.
func convertParamValue(pv, paramName string, config map[string]any) any {
	if strings.ToLower(pv) == "null" {
		return nil
	}
	cfgAny, ok := config[paramName]
	if !ok {
		return pv
	}
	paramType := "string"
	if cfg, ok := cfgAny.(map[string]any); ok {
		if t, ok := cfg["type"]; ok {
			paramType = strings.ToLower(strip(fmt.Sprintf("%v", t)))
		}
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
		if paramType == "object" || strings.HasPrefix(paramType, "dict") {
			if d, err := decodeOrdered(pv); err == nil {
				return d
			}
		}
		if v, ok := pyLiteralEval(pv); ok {
			return v
		}
		return pv
	}
}

// parseXMLFunctionCall turns a "NAME>params" fragment into a tool call.
func (p *SeedOssParser) parseXMLFunctionCall(fcStr string, req Request) *ToolCall {
	endIndex := strings.Index(fcStr, ">")
	if endIndex < 0 {
		return nil
	}
	functionName := fcStr[:endIndex]
	config := seedArgumentsConfig(functionName, req)
	parameters := fcStr[endIndex+1:]
	paramDict := newOmap()
	for _, m := range seedParameterRE.FindAllStringSubmatch(parameters, -1) {
		matchText := m[1]
		if matchText == "" {
			matchText = m[2]
		}
		idx := strings.Index(matchText, ">")
		if idx < 0 {
			continue
		}
		pName := matchText[:idx]
		pValue := matchText[idx+1:]
		pValue = strings.TrimPrefix(pValue, "\n")
		pValue = strings.TrimSuffix(pValue, "\n")
		paramDict.set(pName, convertParamValue(pValue, pName, config))
	}
	return &ToolCall{ID: genToolID(), Name: functionName, Arguments: dumps(paramDict, false)}
}

// seedFunctionCalls returns the "NAME>params" fragments found in the output.
func seedFunctionCalls(modelOutput string) []string {
	var raw []string
	for _, m := range seedToolCallRE.FindAllStringSubmatch(modelOutput, -1) {
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

func (p *SeedOssParser) ExtractToolCalls(modelOutput string, req Request) ExtractedToolCalls {
	if !strings.Contains(modelOutput, seedPrefix) {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	thinkingContent := ""
	resultContent := modelOutput
	if strings.Contains(modelOutput, seedThinkStart) && strings.Contains(modelOutput, seedThinkEnd) {
		thinkEndIndex := strings.Index(modelOutput, seedThinkEnd) + len(seedThinkEnd)
		resultContent = modelOutput[thinkEndIndex:]
		thinkingContent = modelOutput[:thinkEndIndex]
	}

	functionCalls := seedFunctionCalls(resultContent)
	if len(functionCalls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	var calls []ToolCall
	for _, fc := range functionCalls {
		if tc := p.parseXMLFunctionCall(fc, req); tc != nil {
			calls = append(calls, *tc)
		}
	}

	tcStart := strings.Index(resultContent, seedStart)
	if tcStart < 0 {
		tcStart = strings.Index(resultContent, seedPrefix)
	}
	content := thinkingContent + resultContent[:tcStart]

	return ExtractedToolCalls{
		ToolsCalled: len(calls) > 0,
		ToolCalls:   calls,
		Content:     strNil(content),
	}
}

func (p *SeedOssParser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, seedStart) ||
		strings.Contains(text, seedPrefix) ||
		hasTextFormatToolCall(text)
}

func (p *SeedOssParser) ExtractToolCallsStreaming(previousText, currentText, deltaText string, req Request) *StreamDelta {
	if previousText == "" {
		p.Reset()
	}
	if deltaText == "" {
		return nil
	}
	p.accumulatedText = currentText

	// Advance to the next tool once the current one has closed.
	if p.jsonClosed && !p.inFunction {
		toolEnds := strings.Count(currentText, seedEnd)
		if toolEnds > p.currentToolIndex {
			p.currentToolIndex++
			p.headerSent = false
			p.paramCount = 0
			p.jsonStarted = false
			p.jsonClosed = false
			if p.currentToolIndex >= strings.Count(currentText, seedStart) {
				p.isToolCallStarted = false
			}
			return nil
		}
	}

	// Gate on the end of any thinking block.
	if !p.isThinkingEnd {
		if !strings.Contains(currentText, seedThinkStart) || strings.Contains(deltaText, seedThinkEnd) {
			p.isThinkingEnd = true
		}
	}
	if !p.isThinkingEnd {
		return &StreamDelta{Content: ptr(deltaText)}
	}

	// Stream plain content until a tool call starts.
	if !p.isToolCallStarted {
		if strings.Contains(deltaText, seedStart) {
			p.isToolCallStarted = true
			contentBefore := deltaText[:strings.Index(deltaText, seedStart)]
			if contentBefore != "" {
				return &StreamDelta{Content: ptr(contentBefore)}
			}
			// Fall through; the header may already be in current_text.
		} else {
			if strings.HasSuffix(strings.TrimRight(currentText, " \t\n\r\f\v"), seedEnd) && strip(deltaText) == "" {
				return nil
			}
			return &StreamDelta{Content: ptr(deltaText)}
		}
	}

	toolStartsCount := strings.Count(currentText, seedStart)
	if p.currentToolIndex >= toolStartsCount {
		return nil
	}

	// Locate the text of the current tool call.
	thinkEndIdx := 0
	if i := strings.Index(currentText, seedThinkEnd); i >= 0 {
		thinkEndIdx = i + len(seedThinkEnd)
	}
	var toolStarts []int
	idx := thinkEndIdx
	for {
		j := strings.Index(currentText[idx:], seedStart)
		if j < 0 {
			break
		}
		abs := idx + j
		toolStarts = append(toolStarts, abs)
		idx = abs + len(seedStart)
	}
	if p.currentToolIndex >= len(toolStarts) {
		return nil
	}
	toolStartIdx := toolStarts[p.currentToolIndex]
	var toolText string
	if te := strings.Index(currentText[toolStartIdx:], seedEnd); te < 0 {
		toolText = currentText[toolStartIdx:]
	} else {
		toolText = currentText[toolStartIdx : toolStartIdx+te+len(seedEnd)]
	}

	// Parse and emit the function header once.
	if !p.headerSent {
		if strings.Contains(toolText, seedPrefix) {
			funcStart := strings.Index(toolText, seedPrefix) + len(seedPrefix)
			if funcEnd := strings.Index(toolText[funcStart:], ">"); funcEnd != -1 {
				p.currentFunctionName = toolText[funcStart : funcStart+funcEnd]
				p.currentGenID = genToolID()
				p.headerSent = true
				p.inFunction = true

				// If the body is already complete, emit the whole call at once.
				if strings.Contains(toolText, seedFuncEnd) {
					fcEnd := strings.Index(toolText[funcStart:], seedFuncEnd)
					fc := toolText[funcStart : funcStart+fcEnd]
					args := "{}"
					if parsed := p.parseXMLFunctionCall(fc, req); parsed != nil {
						args = parsed.Arguments
					}
					p.jsonStarted = true
					p.jsonClosed = true
					p.inFunction = false
					p.prevToolCallArr = append(p.prevToolCallArr, map[string]any{"name": p.currentFunctionName, "arguments": args})
					return &StreamDelta{ToolCalls: []StreamToolCall{{
						Index:    p.currentToolIndex,
						ID:       ptr(p.currentGenID),
						Type:     ptr(typeFunction),
						Function: &StreamFunction{Name: ptr(p.currentFunctionName), Arguments: ptr(args)},
					}}}
				}

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

		if !p.jsonClosed && strings.Contains(toolText, seedFuncEnd) {
			p.jsonClosed = true
			p.inFunction = false
			funcStart := strings.Index(toolText, seedPrefix) + len(seedPrefix)
			if fcEnd := strings.Index(toolText[funcStart:], seedFuncEnd); fcEnd != -1 {
				fc := toolText[funcStart : funcStart+fcEnd]
				if parsed := p.parseXMLFunctionCall(fc, req); parsed != nil {
					p.prevToolCallArr = append(p.prevToolCallArr, map[string]any{"name": parsed.Name, "arguments": parsed.Arguments})
				}
			}
			return &StreamDelta{ToolCalls: []StreamToolCall{{
				Index:    p.currentToolIndex,
				Function: &StreamFunction{Arguments: ptr("}")},
			}}}
		}

		completeParams := strings.Count(toolText, seedParamEnd)
		if !p.inParam && p.paramCount < completeParams {
			var paramStarts []int
			si := 0
			for {
				j := strings.Index(toolText[si:], seedParamPrefix)
				if j < 0 {
					break
				}
				abs := si + j
				paramStarts = append(paramStarts, abs)
				si = abs + len(seedParamPrefix)
			}
			if len(paramStarts) > p.paramCount {
				paramStart := paramStarts[p.paramCount] + len(seedParamPrefix)
				remaining := toolText[paramStart:]
				if ni := strings.Index(remaining, ">"); ni >= 0 {
					paramName := remaining[:ni]
					valueText := toolText[paramStart+ni+1:]
					valueText = strings.TrimPrefix(valueText, "\n")
					if pe := strings.Index(valueText, seedParamEnd); pe != -1 {
						pv := strings.TrimSuffix(valueText[:pe], "\n")
						config := seedArgumentsConfig(p.currentFunctionName, req)
						serialized := dumps(convertParamValue(pv, paramName, config), false)
						frag := `"` + paramName + `": ` + serialized
						if p.paramCount != 0 {
							frag = ", " + frag
						}
						p.paramCount++
						return &StreamDelta{ToolCalls: []StreamToolCall{{
							Index:    p.currentToolIndex,
							Function: &StreamFunction{Arguments: ptr(frag)},
						}}}
					}
				}
			}
		}
	}

	return nil
}
