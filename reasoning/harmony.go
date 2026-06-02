// SPDX-License-Identifier: Apache-2.0

package reasoning

import (
	"regexp"
	"strings"
)

// Harmony parses the GPT-OSS Harmony channel format:
//
//	<|channel|>analysis<|message|>Let me think...<|end|>
//	<|channel|>final<|message|>The answer is 42.<|return|>
//
// The analysis channel is reasoning; the final channel is content. Commentary
// channels (tool calls) pass through as content for the tool parser.
type Harmony struct {
	currentChannel string
	inMessage      bool
}

// NewHarmony returns a Harmony-format reasoning parser.
func NewHarmony() *Harmony { return &Harmony{} }

var _ ReasoningParser = (*Harmony)(nil)

var (
	harmonyAnalysis = regexp.MustCompile(`(?s)<\|channel\|>analysis\s*<\|message\|>(.*?)<\|end\|>`)
	// Prefer <|return|> termination; fall back to a greedy <|end|> match so a
	// literal <|end|> inside answer text does not truncate the message.
	harmonyFinalReturn = regexp.MustCompile(`(?s)<\|channel\|>final\s*<\|message\|>(.*?)<\|return\|>`)
	harmonyFinalEnd    = regexp.MustCompile(`(?s)<\|channel\|>final\s*<\|message\|>(.*)<\|end\|>`)
)

// ResetState clears streaming channel state.
func (h *Harmony) ResetState() {
	h.currentChannel = ""
	h.inMessage = false
}

// FinalizeStreaming is a no-op for Harmony.
func (h *Harmony) FinalizeStreaming(string) *DeltaMessage { return nil }

// ExtractReasoning collects analysis blocks as reasoning and the final block as
// content.
func (h *Harmony) ExtractReasoning(modelOutput string) (*string, *string) {
	var blocks []string
	for _, m := range harmonyAnalysis.FindAllStringSubmatch(modelOutput, -1) {
		blocks = append(blocks, strings.TrimSpace(m[1]))
	}
	var reasoning *string
	if len(blocks) > 0 {
		reasoning = nz(strings.Join(blocks, "\n"))
	}

	var content *string
	if m := harmonyFinalReturn.FindStringSubmatch(modelOutput); m != nil {
		content = nz(strings.TrimSpace(m[1]))
	} else if m := harmonyFinalEnd.FindStringSubmatch(modelOutput); m != nil {
		content = nz(strings.TrimSpace(m[1]))
	}

	return reasoning, content
}

// ExtractReasoningStreaming tracks the active channel and emits reasoning for
// analysis content, content for final content.
func (h *Harmony) ExtractReasoningStreaming(prev, cur, delta string) *DeltaMessage {
	if strings.Contains(delta, "<|channel|>") {
		switch {
		case strings.Contains(delta, "analysis"):
			h.currentChannel = "analysis"
			h.inMessage = false
			return nil
		case strings.Contains(delta, "final"):
			h.currentChannel = "final"
			h.inMessage = false
			return nil
		case strings.Contains(delta, "commentary"):
			h.currentChannel = "commentary"
			h.inMessage = false
			return &DeltaMessage{Content: &delta}
		}
	}

	if h.currentChannel == "" && strings.Contains(cur, "<|channel|>") {
		after := cur[strings.LastIndex(cur, "<|channel|>")+len("<|channel|>"):]
		switch {
		case strings.HasPrefix(after, "analysis"):
			h.currentChannel = "analysis"
		case strings.HasPrefix(after, "final"):
			h.currentChannel = "final"
		case strings.HasPrefix(after, "commentary"):
			h.currentChannel = "commentary"
		}
	}

	if h.currentChannel == "commentary" {
		return &DeltaMessage{Content: &delta}
	}

	if strings.Contains(delta, "<|message|>") {
		h.inMessage = true
		return nil
	}

	for _, tok := range []string{"<|end|>", "<|return|>", "<|call|>", "<|start|>"} {
		if strings.Contains(delta, tok) {
			h.inMessage = false
			return nil
		}
	}

	if sd := strings.TrimSpace(delta); strings.HasPrefix(sd, "<|") && strings.HasSuffix(sd, "|>") {
		return nil
	}

	if h.inMessage && h.currentChannel == "analysis" {
		return &DeltaMessage{Reasoning: &delta}
	}
	if h.inMessage && h.currentChannel == "final" {
		return &DeltaMessage{Content: &delta}
	}
	return nil
}
