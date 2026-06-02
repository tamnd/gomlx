// SPDX-License-Identifier: Apache-2.0

// Package toolparsers extracts tool calls from model output. Each model family
// emits tool calls in its own wire format, so there is one parser per family
// plus an auto parser that tries several formats in turn. A parser handles both
// a complete response and the streaming case, where output arrives in deltas.
package toolparsers

import (
	"crypto/rand"
	"encoding/hex"
	"regexp"
	"strings"
)

// Request carries the optional request context a parser may consult, most
// importantly the tool definitions used for argument type coercion. It mirrors
// the loosely typed JSON object the HTTP layer holds, so accessors tolerate
// missing or oddly shaped fields.
type Request map[string]any

// tools returns the tool definition objects from the request, or nil.
func (r Request) tools() []map[string]any {
	if r == nil {
		return nil
	}
	raw, ok := r["tools"].([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(raw))
	for _, t := range raw {
		if m, ok := t.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// ToolCall is one extracted call. Arguments is a JSON-encoded string, matching
// the OpenAI tool-call wire shape.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// ExtractedToolCalls is the result of parsing a complete response. Content is
// the leftover text that was not part of any tool call; a nil pointer means the
// content field is absent (as opposed to an empty string).
type ExtractedToolCalls struct {
	ToolsCalled bool
	ToolCalls   []ToolCall
	Content     *string
}

// StreamFunction is the function payload of a streaming tool-call delta. Fields
// are pointers so an absent field serializes away rather than as "".
type StreamFunction struct {
	Name      *string `json:"name,omitempty"`
	Arguments *string `json:"arguments,omitempty"`
}

// StreamToolCall is one tool-call entry in a streaming delta. The first chunk
// for a call carries ID and Type; continuation chunks carry only Index and the
// argument fragment.
type StreamToolCall struct {
	Index    int             `json:"index"`
	ID       *string         `json:"id,omitempty"`
	Type     *string         `json:"type,omitempty"`
	Function *StreamFunction `json:"function,omitempty"`
}

// StreamDelta is what a parser emits for one streaming step. A nil return means
// "emit nothing this step". Content and ToolCalls can both be set on the step
// where text precedes a tool call.
type StreamDelta struct {
	Content   *string          `json:"content,omitempty"`
	ToolCalls []StreamToolCall `json:"tool_calls,omitempty"`
}

// ToolParser parses tool calls from one model family's output.
type ToolParser interface {
	// ExtractToolCalls parses a complete response.
	ExtractToolCalls(modelOutput string, req Request) ExtractedToolCalls
	// ExtractToolCallsStreaming processes one delta using the
	// previous+delta=current model, returning the delta to emit or nil.
	ExtractToolCallsStreaming(previousText, currentText, deltaText string, req Request) *StreamDelta
	// HasPendingToolCall reports whether text holds incomplete tool markup,
	// used to decide whether to keep buffering when a stream ends early.
	HasPendingToolCall(text string) bool
	// Reset clears per-request streaming state.
	Reset()
}

// base holds the small amount of streaming state shared across parsers and
// provides default implementations of the optional interface methods.
type base struct {
	currentToolID   int
	prevToolCallArr []map[string]any
}

func (b *base) Reset() {
	b.currentToolID = -1
	b.prevToolCallArr = nil
}

// ExtractToolCallsStreaming defaults to no streaming support.
func (b *base) ExtractToolCallsStreaming(_, _, _ string, _ Request) *StreamDelta {
	return nil
}

// HasPendingToolCall defaults to the common <tool_call> marker plus the text
// fallback.
func (b *base) HasPendingToolCall(text string) bool {
	return strings.Contains(text, "<tool_call>") || hasTextFormatToolCall(text)
}

// genToolID returns a fresh call id in the common "call_" + 8 hex form.
func genToolID() string {
	var buf [4]byte
	rand.Read(buf[:])
	return "call_" + hex.EncodeToString(buf[:])
}

// ptr returns a pointer to v.
func ptr[T any](v T) *T { return &v }

// strNil returns a pointer to s, or nil when s is empty.
func strNil(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// strip trims surrounding whitespace.
func strip(s string) string { return strings.TrimSpace(s) }

// replaceFirst removes the first match of re from s, mirroring Python's
// re.sub(..., count=1).
func replaceFirst(re *regexp.Regexp, s string) string {
	loc := re.FindStringIndex(s)
	if loc == nil {
		return s
	}
	return s[:loc[0]] + s[loc[1]:]
}

const typeFunction = "function"

// pyStr mirrors Python's str() for the scalar values that show up in tool
// arguments: bool becomes True/False, nil becomes None, a string is itself, and
// numbers print without quotes. Containers fall back to the JSON serializer.
func pyStr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case string:
		return x
	default:
		return dumps(v, false)
	}
}

// argString renders an argument value the way the reference parsers do: a
// JSON object is serialized (honoring asciiOnly), anything else goes through
// Python str() semantics. This matches "json.dumps(args) if isinstance(args,
// dict) else str(args)".
func argString(v any, asciiOnly bool) string {
	if o, ok := v.(*omap); ok {
		return dumps(o, asciiOnly)
	}
	return pyStr(v)
}

// emitAllStreaming turns a complete extraction into a streaming delta that
// emits every call as one entry starting at index 0. Parsers that re-extract
// the full text whenever a closing marker arrives share this shape.
func emitAllStreaming(res ExtractedToolCalls) *StreamDelta {
	if !res.ToolsCalled {
		return nil
	}
	return streamFromCalls(res.ToolCalls, 0)
}

// --- think-tag stripping --------------------------------------------------

var (
	thinkTagRE      = regexp.MustCompile(`(?s)<think>.*?</think>`)
	implicitThinkRE = regexp.MustCompile(`(?s)^.*?</think>`)
)

// stripThinkTags removes think markup so tool parsing is not confused by a
// model that emits reasoning. It handles both full <think>...</think> blocks
// and the implicit case where only the closing tag appears (the opening tag was
// in the prompt). The result is trimmed.
func stripThinkTags(text string) string {
	result := thinkTagRE.ReplaceAllString(text, "")
	if result == text && strings.Contains(text, "</think>") {
		result = implicitThinkRE.ReplaceAllString(text, "")
	}
	return strip(result)
}

// --- text-format tool-call fallback ---------------------------------------
//
// At low quantization, some models drop out of their structured format after a
// few tool rounds and emit calls as plain text. These helpers recover the two
// common shapes:
//
//	[Calling tool="name" key="value" ...]
//	[Calling tool: name({json})]

var (
	textKVPattern = regexp.MustCompile(`\[Calling\s+tool="([^"]+)"((?:\s+\w+="(?:[^"\\]|\\.)*")*)\s*\]`)
	textKVParam   = regexp.MustCompile(`(\w+)="((?:[^"\\]|\\.)*)"`)
	textFnPattern = regexp.MustCompile(`(?s)\[Calling\s+tool:\s*(\w+)\((\{.*?\})\)\s*\]`)
	textAny       = regexp.MustCompile(`\[Calling\s+tool[=:]`)
)

// hasTextFormatToolCall reports whether text contains either text-format shape.
func hasTextFormatToolCall(text string) bool {
	return textAny.MatchString(text)
}

// extractTextFormatToolCalls recovers tool calls from the text-format shapes.
func extractTextFormatToolCalls(text string) []ToolCall {
	var calls []ToolCall

	for _, m := range textKVPattern.FindAllStringSubmatch(text, -1) {
		funcName, paramsStr := m[1], m[2]
		args := newOmap()
		for _, pm := range textKVParam.FindAllStringSubmatch(paramsStr, -1) {
			key := pm[1]
			value := strings.ReplaceAll(pm[2], `\"`, `"`)
			if decoded, err := decodeOrdered(value); err == nil {
				args.set(key, decoded)
			} else {
				args.set(key, value)
			}
		}
		if args.len() > 0 {
			calls = append(calls, ToolCall{
				ID:        genToolID(),
				Name:      strip(funcName),
				Arguments: dumps(args, false),
			})
		}
	}

	for _, m := range textFnPattern.FindAllStringSubmatch(text, -1) {
		funcName, jsonStr := m[1], m[2]
		decoded, err := decodeOrdered(jsonStr)
		if err != nil {
			continue
		}
		if o, ok := decoded.(*omap); ok && o.len() > 0 {
			calls = append(calls, ToolCall{
				ID:        genToolID(),
				Name:      strip(funcName),
				Arguments: dumps(o, false),
			})
		}
	}

	return calls
}

// streamFromCalls builds a streaming delta that emits the given calls as
// complete entries, starting at startIndex. This is the shape used by parsers
// that re-extract on each closing marker.
func streamFromCalls(calls []ToolCall, startIndex int) *StreamDelta {
	if len(calls) == 0 {
		return nil
	}
	entries := make([]StreamToolCall, len(calls))
	for i, tc := range calls {
		entries[i] = StreamToolCall{
			Index:    startIndex + i,
			ID:       ptr(tc.ID),
			Type:     ptr(typeFunction),
			Function: &StreamFunction{Name: ptr(tc.Name), Arguments: ptr(tc.Arguments)},
		}
	}
	return &StreamDelta{ToolCalls: entries}
}
