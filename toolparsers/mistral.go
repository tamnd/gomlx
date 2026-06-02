// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"crypto/rand"
	"regexp"
	"strings"
)

const mistralBotToken = "[TOOL_CALLS]"

// Old Mistral output is a JSON array after the marker; this recovers it when
// the array is embedded in surrounding text.
var mistralArrayRE = regexp.MustCompile(`(?s)\[{.*}\]`)

const mistralIDAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

// genMistralToolID returns a 9-character alphanumeric id, the shape Mistral
// expects for tool-call ids.
func genMistralToolID() string {
	var buf [9]byte
	rand.Read(buf[:])
	for i := range buf {
		buf[i] = mistralIDAlphabet[int(buf[i])%len(mistralIDAlphabet)]
	}
	return string(buf[:])
}

// MistralParser handles both the old array form and the newer
// name{json} form, including Devstral's [ARGS] separator.
type MistralParser struct{ base }

func (p *MistralParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	if !strings.Contains(modelOutput, mistralBotToken) {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}

	parts := strings.Split(modelOutput, mistralBotToken)
	content := strip(parts[0])
	rawCalls := parts[1:]

	var calls []ToolCall
	for _, raw := range rawCalls {
		raw = strip(raw)
		if raw == "" {
			continue
		}

		// New format: name{json}, possibly name[ARGS]{json} from Devstral.
		if !strings.HasPrefix(raw, "[") && strings.Contains(raw, "{") {
			endName := strings.Index(raw, "{")
			name := strip(strings.ReplaceAll(raw[:endName], "[ARGS]", ""))
			argsStr := raw[endName:]
			if name != "" {
				calls = append(calls, ToolCall{ID: genMistralToolID(), Name: name, Arguments: argsStr})
			}
			continue
		}

		// Old format: a JSON array of calls.
		if decoded, err := decodeOrdered(raw); err == nil {
			if arr, ok := decoded.([]any); ok {
				calls = append(calls, mistralCallsFromArray(arr)...)
			}
			continue
		}

		// Fallback: pull a JSON array out of surrounding text.
		if m := mistralArrayRE.FindString(raw); m != "" {
			if decoded, err := decodeOrdered(m); err == nil {
				if arr, ok := decoded.([]any); ok {
					calls = append(calls, mistralCallsFromArray(arr)...)
					continue
				}
			}
		}
		// If nothing parsed, fold the text back into content.
		if content != "" {
			content = strip(content + " " + raw)
		} else {
			content = raw
		}
	}

	if len(calls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(content)}
}

// mistralCallsFromArray turns an old-format array into tool calls.
func mistralCallsFromArray(arr []any) []ToolCall {
	var calls []ToolCall
	for _, item := range arr {
		obj, ok := item.(*omap)
		if !ok || !obj.has("name") {
			continue
		}
		name, _ := obj.get("name").(string)
		args := obj.get("arguments")
		if args == nil {
			args = newOmap()
		}
		calls = append(calls, ToolCall{ID: genMistralToolID(), Name: name, Arguments: argString(args, false)})
	}
	return calls
}

func (p *MistralParser) ExtractToolCallsStreaming(_, currentText, deltaText string, _ Request) *StreamDelta {
	if !strings.Contains(currentText, mistralBotToken) {
		return &StreamDelta{Content: ptr(deltaText)}
	}

	// The delta that introduces the marker: split off any leading content, then
	// open a fresh tool call.
	if strings.Contains(deltaText, mistralBotToken) {
		parts := strings.SplitN(deltaText, mistralBotToken, 2)
		contentPart := parts[0]
		toolPart := parts[1]

		delta := &StreamDelta{}
		emit := false
		if contentPart != "" {
			delta.Content = ptr(contentPart)
			emit = true
		}
		p.currentToolID++
		if toolPart != "" {
			if fn := parseMistralToolDelta(toolPart); fn != nil {
				delta.ToolCalls = []StreamToolCall{{
					Index:    p.currentToolID,
					ID:       ptr(genMistralToolID()),
					Type:     ptr(typeFunction),
					Function: fn,
				}}
				emit = true
			}
		}
		if emit {
			return delta
		}
		return nil
	}

	// Continuation of an open tool call.
	if p.currentToolID >= 0 {
		if fn := parseMistralToolDelta(deltaText); fn != nil {
			return &StreamDelta{ToolCalls: []StreamToolCall{{
				Index:    p.currentToolID,
				Type:     ptr(typeFunction),
				Function: fn,
			}}}
		}
	}
	return nil
}

// parseMistralToolDelta splits a streaming fragment into a name and/or
// arguments piece, mirroring the reference incremental parser.
func parseMistralToolDelta(text string) *StreamFunction {
	if text == "" {
		return nil
	}
	fn := &StreamFunction{}
	set := false
	if i := strings.Index(text, "{"); i >= 0 {
		namePart := text[:i]
		argsPart := text[i:]
		if strip(namePart) != "" {
			fn.Name = ptr(strip(strings.ReplaceAll(namePart, "[ARGS]", "")))
			set = true
		}
		if argsPart != "" {
			fn.Arguments = ptr(argsPart)
			set = true
		}
	} else if strip(text) != "" && !strings.ContainsAny(text[:1], "{}[],") {
		fn.Name = ptr(strip(text))
		set = true
	} else {
		fn.Arguments = ptr(text)
		set = true
	}
	if !set {
		return nil
	}
	return fn
}

func (p *MistralParser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, mistralBotToken)
}
