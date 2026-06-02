// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

var (
	// Header before the JSON args in the recipient form. The args themselves
	// are scanned by hand below, because RE2 has no lookahead and the original
	// pattern asserts the closing brace sits right before "<|" or end of text.
	functionaryRecipientHeadRE = regexp.MustCompile(`(?s)^\s*(\w+)\s*\n<\|content\|>\s*`)
	functionaryFromRE          = regexp.MustCompile(`<\|from\|>assistant\s*`)
	functionaryArrayRE         = regexp.MustCompile(`(?s)^\s*\[.*\]\s*$`)
)

const functionaryRecipientTok = "<|recipient|>"

// FunctionaryParser handles MeetKai Functionary output: a recipient/content
// header form, a bare <function=> form, and an OpenAI-style JSON array.
type FunctionaryParser struct{ base }

// recipientMatch is one parsed recipient block plus the span it covers in the
// source, so the caller can strip it the way the reference regex sub does.
type recipientMatch struct {
	name  string
	args  string
	start int
	end   int
}

// findRecipientMatches reproduces the recipient regex including its lookahead:
// each match is "<|recipient|> name \n <|content|> {json}" where the closing
// brace is immediately followed by "<|" or the end of the text.
func findRecipientMatches(text string) []recipientMatch {
	var out []recipientMatch
	for i := 0; ; {
		rel := strings.Index(text[i:], functionaryRecipientTok)
		if rel < 0 {
			break
		}
		start := i + rel
		rest := text[start+len(functionaryRecipientTok):]
		head := functionaryRecipientHeadRE.FindStringSubmatchIndex(rest)
		if head == nil {
			i = start + len(functionaryRecipientTok)
			continue
		}
		name := rest[head[2]:head[3]]
		argsStart := start + len(functionaryRecipientTok) + head[1]
		if argsStart >= len(text) || text[argsStart] != '{' {
			i = start + len(functionaryRecipientTok)
			continue
		}
		// Shortest {...} whose closing brace borders "<|" or end of text.
		closeIdx := -1
		for j := argsStart; j < len(text); j++ {
			if text[j] != '}' {
				continue
			}
			after := text[j+1:]
			if after == "" || after == "\n" || strings.HasPrefix(after, "<|") {
				closeIdx = j
				break
			}
		}
		if closeIdx < 0 {
			i = start + len(functionaryRecipientTok)
			continue
		}
		out = append(out, recipientMatch{
			name:  name,
			args:  text[argsStart : closeIdx+1],
			start: start,
			end:   closeIdx + 1,
		})
		i = closeIdx + 1
	}
	return out
}

// removeSpans returns text with the given (start,end) ranges deleted.
func removeSpans(text string, matches []recipientMatch) string {
	var b strings.Builder
	prev := 0
	for _, m := range matches {
		b.WriteString(text[prev:m.start])
		prev = m.end
	}
	b.WriteString(text[prev:])
	return b.String()
}

func (p *FunctionaryParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	var calls []ToolCall
	cleaned := modelOutput

	recipients := findRecipientMatches(modelOutput)
	for _, m := range recipients {
		lower := strings.ToLower(m.name)
		if lower == "all" || lower == "user" {
			continue
		}
		calls = append(calls, ToolCall{ID: genToolID(), Name: m.name, Arguments: m.args})
	}
	if len(recipients) > 0 {
		cleaned = removeSpans(cleaned, recipients)
		cleaned = strip(functionaryFromRE.ReplaceAllString(cleaned, ""))
	}

	fnMatches := llamaFunctionRE.FindAllStringSubmatch(cleaned, -1)
	for _, m := range fnMatches {
		calls = append(calls, ToolCall{ID: genToolID(), Name: strip(m[1]), Arguments: m[2]})
	}
	if len(fnMatches) > 0 {
		cleaned = strip(llamaFunctionRE.ReplaceAllString(cleaned, ""))
	}

	// OpenAI-style JSON array, only when nothing else matched.
	if len(calls) == 0 && functionaryArrayRE.MatchString(strip(modelOutput)) {
		if decoded, err := decodeOrdered(strip(modelOutput)); err == nil {
			if arr, ok := decoded.([]any); ok {
				for _, item := range arr {
					call, ok := item.(*omap)
					if !ok || !call.has("name") {
						continue
					}
					name, _ := call.get("name").(string)
					args := call.get("arguments")
					if args == nil {
						args = newOmap()
					}
					calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(args, false)})
				}
				cleaned = ""
			}
		}
	}

	if len(calls) == 0 {
		return ExtractedToolCalls{Content: ptr(modelOutput)}
	}
	return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: strNil(cleaned)}
}

func (p *FunctionaryParser) ExtractToolCallsStreaming(_, currentText, deltaText string, req Request) *StreamDelta {
	if !strings.Contains(currentText, "<|recipient|>") &&
		!strings.Contains(currentText, "<function=") &&
		!strings.Contains(currentText, "[") {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	if strings.Contains(deltaText, "<|content|>") ||
		strings.Contains(deltaText, "</function>") ||
		strings.Contains(deltaText, "]") {
		return emitAllStreaming(p.ExtractToolCalls(currentText, req))
	}
	return nil
}
