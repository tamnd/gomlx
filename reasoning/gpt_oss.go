// SPDX-License-Identifier: Apache-2.0

package reasoning

import (
	"regexp"
	"strings"
)

// GptOss parses the GPT-OSS channel format:
//
//	<|channel|>analysis<|message|>[reasoning]<|start|>assistant<|channel|>final<|message|>[content]<|return|>
//
// The 'analysis' channel maps to reasoning, 'final' to content. An extended
// form with <|constrain|>JSON before <|message|> is also accepted.
type GptOss struct{}

// NewGptOss returns a GPT-OSS (channel-format) reasoning parser.
func NewGptOss() *GptOss { return &GptOss{} }

var _ ReasoningParser = (*GptOss)(nil)

var (
	gptStructural = regexp.MustCompile(`<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>|<\|constrain\|>`)
	gptChannelRE  = regexp.MustCompile(`<\|channel\|>(analysis|final)(?:[^<]*(?:<\|constrain\|>[^<]*)?)?<\|message\|>`)
)

// ResetState is a no-op: GPT-OSS streaming is stateless (phase is derived from
// the accumulated text on each call).
func (GptOss) ResetState() {}

// FinalizeStreaming is a no-op for GPT-OSS.
func (GptOss) FinalizeStreaming(string) *DeltaMessage { return nil }

// gptExtractChannel returns the content of the first named channel, or nil.
func gptExtractChannel(text, channel string) *string {
	for _, m := range gptChannelRE.FindAllStringSubmatchIndex(text, -1) {
		// m[2]:m[3] is group 1 (channel name); m[1] is the match end.
		if text[m[2]:m[3]] != channel {
			continue
		}
		start := m[1]
		rest := text[start:]
		if loc := gptStructural.FindStringIndex(rest); loc != nil {
			return stripNil(rest[:loc[0]])
		}
		return stripNil(rest)
	}
	return nil
}

// ExtractReasoning splits complete output into (reasoning, content).
func (GptOss) ExtractReasoning(modelOutput string) (*string, *string) {
	if modelOutput == "" || !strings.Contains(modelOutput, "<|channel|>") {
		return nil, nz(modelOutput)
	}

	reasoning := gptExtractChannel(modelOutput, "analysis")
	content := gptExtractChannel(modelOutput, "final")

	if content != nil {
		c := strings.TrimSpace(strings.ReplaceAll(*content, "<|return|>", ""))
		c = strings.TrimSpace(gptStructural.ReplaceAllString(c, ""))
		content = nz(c)
	}
	if reasoning != nil {
		reasoning = nz(strings.TrimSpace(gptStructural.ReplaceAllString(*reasoning, "")))
	}

	if reasoning == nil && content == nil {
		return nil, &modelOutput
	}
	return reasoning, content
}

// ExtractReasoningStreaming derives the phase from the accumulated text.
func (GptOss) ExtractReasoningStreaming(prev, cur, delta string) *DeltaMessage {
	prevPhase := gptDetectPhase(prev)
	currPhase := gptDetectPhase(cur)

	if currPhase != prevPhase && (currPhase == "analysis" || currPhase == "final") {
		after := gptContentAfterMarker(cur, currPhase)
		if after != "" {
			after = strings.ReplaceAll(after, "<|return|>", "")
			if currPhase == "analysis" {
				return &DeltaMessage{Reasoning: nz(after)}
			}
			return &DeltaMessage{Content: nz(after)}
		}
		return nil
	}

	switch currPhase {
	case "analysis":
		cleaned := gptStructural.ReplaceAllString(strings.ReplaceAll(delta, "<|return|>", ""), "")
		return msgIfNonEmpty(cleaned, true)
	case "final":
		cleaned := gptStructural.ReplaceAllString(strings.ReplaceAll(delta, "<|return|>", ""), "")
		return msgIfNonEmpty(cleaned, false)
	}
	return nil
}

func msgIfNonEmpty(s string, reasoning bool) *DeltaMessage {
	if s == "" {
		return nil
	}
	if reasoning {
		return &DeltaMessage{Reasoning: &s}
	}
	return &DeltaMessage{Content: &s}
}

// gptDetectPhase reports the streaming phase implied by accumulated text:
// "final", "analysis", "transition", or "init".
func gptDetectPhase(text string) string {
	matches := gptChannelRE.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return "init"
	}
	last := matches[len(matches)-1]
	if text[last[2]:last[3]] == "final" {
		return "final"
	}
	after := text[last[1]:]
	if gptStructural.MatchString(after) {
		return "transition"
	}
	return "analysis"
}

// gptContentAfterMarker returns the text following the last marker of the given
// phase's channel.
func gptContentAfterMarker(cur, phase string) string {
	matches := gptChannelRE.FindAllStringSubmatchIndex(cur, -1)
	for i := len(matches) - 1; i >= 0; i-- {
		m := matches[i]
		if cur[m[2]:m[3]] == phase {
			return cur[m[1]:]
		}
	}
	return ""
}
