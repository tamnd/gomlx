// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"regexp"
	"strings"
)

// GPT-OSS Harmony output uses control tokens and channels. Tool calls live in
// the commentary channel addressed to=functions.NAME; the user-facing reply is
// in the final channel. The commentary pattern accepts either the model-
// generated or the template-encoded ordering, and treats a missing terminator
// (end of text or the next channel marker) as a valid end so a complete but
// unterminated call still parses.
var (
	harmonyCommentaryRE = regexp.MustCompile(`(?s)(?:` +
		`to=functions\.([\w-]+)<\|channel\|>commentary(?:\s+\w+)?<\|message\|>(.*?)` +
		`(?:<\|call\|>|<\|channel\|>|<\|start\|>|<\|end\|>|<\|return\|>|$)` +
		`|` +
		`<\|channel\|>commentary\s+to=functions\.([\w-]+)(?:\s*<\|constrain\|>\w+)?\s*<\|message\|>(.*?)` +
		`(?:<\|call\|>|<\|channel\|>|<\|start\|>|<\|end\|>|<\|return\|>|$)` +
		`)`)
	harmonyFinalRE = regexp.MustCompile(`(?s)<\|channel\|>final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)`)

	harmonyChannelNameRE = regexp.MustCompile(`(?:analysis|commentary|final)\s*`)
	harmonyToFuncRE      = regexp.MustCompile(`to=functions\.\w+\s*`)
	harmonyJSONWordRE    = regexp.MustCompile(`json\s*`)
)

var harmonyControlTokens = []string{
	"<|start|>", "<|end|>", "<|message|>", "<|channel|>",
	"<|constrain|>", "<|return|>", "<|call|>",
}

// stripControlTokens removes Harmony control tokens and channel bookkeeping.
func stripControlTokens(text string) string {
	for _, tok := range harmonyControlTokens {
		text = strings.ReplaceAll(text, tok, "")
	}
	text = harmonyChannelNameRE.ReplaceAllString(text, "")
	text = harmonyToFuncRE.ReplaceAllString(text, "")
	text = harmonyJSONWordRE.ReplaceAllString(text, "")
	return strip(text)
}

// HarmonyParser handles the GPT-OSS Harmony channel format.
type HarmonyParser struct{ base }

func (p *HarmonyParser) ExtractToolCalls(modelOutput string, _ Request) ExtractedToolCalls {
	var calls []ToolCall
	for _, m := range harmonyCommentaryRE.FindAllStringSubmatch(modelOutput, -1) {
		name := m[1]
		if name == "" {
			name = m[3]
		}
		argsStr := m[2]
		if argsStr == "" {
			argsStr = m[4]
		}
		argsStr = strip(argsStr)

		if decoded, err := decodeOrdered(argsStr); err == nil {
			calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argString(decoded, false)})
		} else {
			calls = append(calls, ToolCall{ID: genToolID(), Name: name, Arguments: argsStr})
		}
	}

	var content *string
	if m := harmonyFinalRE.FindStringSubmatch(modelOutput); m != nil {
		content = ptr(strip(m[1]))
	}

	if len(calls) > 0 {
		return ExtractedToolCalls{ToolsCalled: true, ToolCalls: calls, Content: content}
	}

	if content == nil {
		content = ptr(stripControlTokens(modelOutput))
	}
	return ExtractedToolCalls{Content: content}
}

func (p *HarmonyParser) ExtractToolCallsStreaming(previousText, currentText, deltaText string, req Request) *StreamDelta {
	if strings.Contains(deltaText, "<|call|>") {
		result := p.ExtractToolCalls(currentText, req)
		if result.ToolsCalled {
			return streamFromCalls(result.ToolCalls, 0)
		}
	}

	if strings.Contains(currentText, "<|channel|>final") {
		finalStart := strings.LastIndex(currentText, "<|channel|>final")
		msgStart := strings.Index(currentText[finalStart:], "<|message|>")
		if msgStart >= 0 {
			msgStart += finalStart
			raw := currentText[msgStart+len("<|message|>"):]
			clean := stripControlTokens(raw)

			prevClean := ""
			if prevFinal := strings.LastIndex(previousText, "<|channel|>final"); prevFinal >= 0 {
				if prevMsg := strings.Index(previousText[prevFinal:], "<|message|>"); prevMsg >= 0 {
					prevMsg += prevFinal
					prevClean = stripControlTokens(previousText[prevMsg+len("<|message|>"):])
				}
			}

			// Emit only the runes added since the previous extraction.
			r := []rune(clean)
			pr := len([]rune(prevClean))
			if pr < len(r) {
				if newContent := string(r[pr:]); newContent != "" {
					return &StreamDelta{Content: ptr(newContent)}
				}
			}
		}
		return &StreamDelta{Content: ptr("")}
	}

	if !strings.Contains(currentText, "<|channel|>") {
		return &StreamDelta{Content: ptr(deltaText)}
	}
	return nil
}

func (p *HarmonyParser) HasPendingToolCall(text string) bool {
	return strings.Contains(text, "to=functions.")
}
