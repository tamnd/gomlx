// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// DeepSeek-V3.1 wraps each call between begin/end markers with a separator
// between the function name and its JSON arguments. Unlike the V3 format there
// is no ```json fence around the body.
const (
	dsv31CallsStart = "<｜tool▁calls▁begin｜>"
	dsv31CallsEnd   = "<｜tool▁calls▁end｜>"
	dsv31CallStart  = "<｜tool▁call▁begin｜>"
	dsv31CallEnd    = "<｜tool▁call▁end｜>"
)

var (
	dsv31CallRE       = regexp.MustCompile(`(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>`)
	dsv31PortionRE    = regexp.MustCompile(`(?s)(.*)<｜tool▁sep｜>(.*)`)
	dsv31PortionPreRE = regexp.MustCompile(`(?s)(.*)<｜tool▁sep｜>`)
)

// DeepSeekV31Parser handles the DeepSeek-V3.1 tool-call format. Its streaming
// path diffs the growing argument string against what was already emitted, the
// same incremental contract the reference parser implements.
type DeepSeekV31Parser struct {
	base
	currentToolNameSent bool
	streamedArgsForTool []string
}

func (p *DeepSeekV31Parser) Reset() {
	p.base.Reset()
	p.currentToolNameSent = false
	p.streamedArgsForTool = nil
}

func (p *DeepSeekV31Parser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	if !strings.Contains(modelOutput, dsv31CallsStart) {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	var calls []ToolCall
	for _, m := range dsv31CallRE.FindAllStringSubmatch(modelOutput, -1) {
		calls = append(calls, ToolCall{
			ID:        genToolID(),
			Name:      strip(m[1]),
			Arguments: strip(m[2]),
		})
	}

	content := modelOutput[:strings.Index(modelOutput, dsv31CallsStart)]
	return ExtractedToolCalls{
		ToolsCalled: len(calls) > 0,
		ToolCalls:   calls,
		Content:     strNil(content),
	}
}

func (p *DeepSeekV31Parser) ExtractToolCallsStreaming(previousText, currentText, deltaText string, _ Request) *StreamDelta {
	if previousText == "" {
		p.currentToolID = -1
		p.currentToolNameSent = false
		p.streamedArgsForTool = nil
		p.prevToolCallArr = nil
	}

	if !strings.Contains(currentText, dsv31CallsStart) {
		return &StreamDelta{Content: ptr(deltaText)}
	}

	// Strip the calls-block markers from anything we might emit as content.
	deltaText = strings.ReplaceAll(deltaText, dsv31CallsStart, "")
	deltaText = strings.ReplaceAll(deltaText, dsv31CallsEnd, "")

	prevStart := strings.Count(previousText, dsv31CallStart)
	prevEnd := strings.Count(previousText, dsv31CallEnd)
	curStart := strings.Count(currentText, dsv31CallStart)
	curEnd := strings.Count(currentText, dsv31CallEnd)

	// Inside a settled region with no new call boundary: pass text through.
	if curStart == curEnd && prevEnd == curEnd && !strings.Contains(deltaText, dsv31CallEnd) {
		return &StreamDelta{Content: ptr(deltaText)}
	}

	var toolCallPortion *string
	if strings.Contains(deltaText, dsv31CallEnd) {
		portion := strip(lastSplit(currentText, dsv31CallStart))
		portion = firstSplit(portion, dsv31CallEnd)
		portion = strings.TrimRight(portion, " \t\n\r\f\v")
		toolCallPortion = &portion
		deltaText = strings.TrimRight(firstSplit(deltaText, dsv31CallEnd), " \t\n\r\f\v")
	}

	switch {
	case curStart > curEnd && curStart > prevStart:
		// A new tool call opened in this delta.
		if len(deltaText) > 1 {
			portion := lastSplit(currentText, dsv31CallStart)
			toolCallPortion = &portion
		} else {
			toolCallPortion = nil
		}
		p.currentToolID++
		p.currentToolNameSent = false
		p.streamedArgsForTool = append(p.streamedArgsForTool, "")

	case curStart > curEnd && curStart == prevStart:
		// Still filling the current open call.
		portion := lastSplit(currentText, dsv31CallStart)
		toolCallPortion = &portion

	case curStart == curEnd && curEnd >= prevEnd:
		// The current call just closed: flush the tail of its arguments.
		if len(p.prevToolCallArr) == 0 || p.currentToolID < 0 || p.currentToolID >= len(p.prevToolCallArr) {
			return nil
		}
		diff, _ := p.prevToolCallArr[p.currentToolID]["arguments"].(string)
		if diff != "" && strings.Contains(deltaText, `"}`) {
			endLoc := strings.LastIndex(deltaText, `"}`)
			diff = deltaText[:endLoc] + `"}`
			p.streamedArgsForTool[p.currentToolID] += diff
			return &StreamDelta{ToolCalls: []StreamToolCall{{
				Index:    p.currentToolID,
				Function: &StreamFunction{Arguments: ptr(diff)},
			}}}
		}
		return nil

	default:
		text := strings.ReplaceAll(deltaText, dsv31CallsStart, "")
		text = strings.ReplaceAll(text, dsv31CallsEnd, "")
		if text != "" {
			return &StreamDelta{Content: ptr(text)}
		}
		return nil
	}

	// Parse the portion into a name and (partial) arguments.
	currentToolCall := map[string]any{}
	if toolCallPortion != nil {
		if m := dsv31PortionRE.FindStringSubmatch(*toolCallPortion); m != nil {
			currentToolCall["name"] = strip(m[1])
			currentToolCall["arguments"] = m[2]
		} else if m := dsv31PortionPreRE.FindStringSubmatch(*toolCallPortion); m != nil {
			currentToolCall["name"] = strip(m[1])
			currentToolCall["arguments"] = ""
		} else {
			return nil
		}
	}

	// Send the function name once, before any argument fragments.
	if !p.currentToolNameSent {
		if len(currentToolCall) == 0 {
			return nil
		}
		funcName, _ := currentToolCall["name"].(string)
		if funcName != "" {
			p.currentToolNameSent = true
			return &StreamDelta{ToolCalls: []StreamToolCall{{
				Index:    p.currentToolID,
				ID:       ptr(genToolID()),
				Type:     ptr(typeFunction),
				Function: &StreamFunction{Name: ptr(funcName), Arguments: ptr("")},
			}}}
		}
		return nil
	}

	if toolCallPortion == nil {
		return nil
	}

	if len(p.prevToolCallArr) <= p.currentToolID {
		p.prevToolCallArr = append(p.prevToolCallArr, map[string]any{})
	}
	prevArgs, _ := p.prevToolCallArr[p.currentToolID]["arguments"].(string)
	curArgs, _ := currentToolCall["arguments"].(string)

	var delta *StreamDelta
	switch {
	case curArgs == "" && prevArgs == "":
		delta = nil
	case curArgs != "" && prevArgs == "":
		delta = &StreamDelta{ToolCalls: []StreamToolCall{{
			Index:    p.currentToolID,
			Function: &StreamFunction{Arguments: ptr(curArgs)},
		}}}
		p.streamedArgsForTool[p.currentToolID] = curArgs
	case curArgs != "" && prevArgs != "":
		if len(curArgs) > len(prevArgs) && strings.HasPrefix(curArgs, prevArgs) {
			diff := curArgs[len(prevArgs):]
			delta = &StreamDelta{ToolCalls: []StreamToolCall{{
				Index:    p.currentToolID,
				Function: &StreamFunction{Arguments: ptr(diff)},
			}}}
			p.streamedArgsForTool[p.currentToolID] = curArgs
		}
	}

	if p.currentToolID == len(p.prevToolCallArr)-1 {
		p.prevToolCallArr[p.currentToolID] = currentToolCall
	} else {
		p.prevToolCallArr = append(p.prevToolCallArr, currentToolCall)
	}
	return delta
}

func (p *DeepSeekV31Parser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, dsv31CallStart)
}

// lastSplit returns the segment of s after the final occurrence of sep, or s
// when sep is absent. It mirrors Python's str.split(sep)[-1].
func lastSplit(s, sep string) string {
	if i := strings.LastIndex(s, sep); i >= 0 {
		return s[i+len(sep):]
	}
	return s
}

// firstSplit returns the segment of s before the first occurrence of sep, or s
// when sep is absent. It mirrors Python's str.split(sep)[0].
func firstSplit(s, sep string) string {
	if i := strings.Index(s, sep); i >= 0 {
		return s[:i]
	}
	return s
}
