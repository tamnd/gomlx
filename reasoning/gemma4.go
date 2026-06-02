// SPDX-License-Identifier: Apache-2.0

package reasoning

import (
	"regexp"
	"strings"
)

// Gemma4 separates thinking from content using channel tokens:
//
//	<|channel>thought\n...reasoning...<channel|>
//	<|channel>content\n...answer...<channel|>
type Gemma4 struct {
	inThought     bool
	inContent     bool
	sawAnyChannel bool
}

// NewGemma4 returns a Gemma 4 reasoning parser.
func NewGemma4() *Gemma4 { return &Gemma4{} }

var _ ReasoningParser = (*Gemma4)(nil)

var (
	gemmaThoughtBlock = regexp.MustCompile(`<\|channel>thought\n[\s\S]*?<channel\|>\s*`)
	gemmaContentStart = regexp.MustCompile(`<\|channel>(?:content|final)\n?`)
	gemmaChannelEnd   = regexp.MustCompile(`<channel\|>`)
	gemmaTurnEnd      = regexp.MustCompile(`<turn\|>`)
)

// ResetState clears streaming channel state.
func (g *Gemma4) ResetState() {
	g.inThought = false
	g.inContent = false
	g.sawAnyChannel = false
}

// ExtractReasoning extracts thought blocks as reasoning and the rest as content.
func (g *Gemma4) ExtractReasoning(modelOutput string) (*string, *string) {
	if modelOutput == "" {
		return nil, nz(modelOutput)
	}

	blocks := gemmaThoughtBlock.FindAllString(modelOutput, -1)
	if len(blocks) == 0 {
		cleaned := gemmaContentStart.ReplaceAllString(modelOutput, "")
		cleaned = gemmaChannelEnd.ReplaceAllString(cleaned, "")
		cleaned = strings.TrimSpace(gemmaTurnEnd.ReplaceAllString(cleaned, ""))
		return nil, &cleaned
	}

	var reasoning strings.Builder
	for _, block := range blocks {
		inner := strings.ReplaceAll(block, "<|channel>thought\n", "")
		inner = strings.ReplaceAll(inner, "<channel|>", "")
		reasoning.WriteString(strings.TrimSpace(inner))
	}

	content := gemmaThoughtBlock.ReplaceAllString(modelOutput, "")
	content = gemmaContentStart.ReplaceAllString(content, "")
	content = gemmaChannelEnd.ReplaceAllString(content, "")
	content = strings.TrimSpace(gemmaTurnEnd.ReplaceAllString(content, ""))

	return nz(reasoning.String()), nz(content)
}

// ExtractReasoningStreaming tracks the active channel across deltas.
func (g *Gemma4) ExtractReasoningStreaming(prev, cur, delta string) *DeltaMessage {
	if delta == "" {
		return nil
	}

	wasInThought := g.inThought

	if strings.Contains(cur, "<|channel>thought") && !g.inContent {
		g.inThought = true
		g.sawAnyChannel = true
	}
	if strings.Contains(cur, "<|channel>content") || strings.Contains(cur, "<|channel>final") {
		g.inThought = false
		g.inContent = true
	}
	if g.inThought && strings.Contains(cur, "<channel|>") {
		thoughtStarts := strings.Count(cur, "<|channel>thought")
		channelEnds := strings.Count(cur, "<channel|>")
		if channelEnds >= thoughtStarts {
			g.inThought = false
			if !strings.Contains(cur, "<|channel>content") && !strings.Contains(cur, "<|channel>final") {
				g.inContent = true
			}
		}
	}

	// A thought-to-content flip during this delta: split so reasoning bytes
	// before the marker stay in reasoning rather than leaking into content.
	if wasInThought && !g.inThought {
		flipPos := -1
		for _, marker := range []string{"<channel|>", "<|channel>content", "<|channel>final"} {
			if idx := strings.Index(delta, marker); idx >= 0 && (flipPos < 0 || idx < flipPos) {
				flipPos = idx
			}
		}
		if flipPos >= 0 {
			pre := gemmaStripMarkers(delta[:flipPos])
			post := gemmaStripMarkers(delta[flipPos:])
			if pre != "" || post != "" {
				return &DeltaMessage{Reasoning: nz(pre), Content: nz(post)}
			}
		}
	}

	clean := gemmaStripMarkers(delta)
	if clean == "" {
		return nil
	}

	switch {
	case g.inThought:
		return &DeltaMessage{Reasoning: &clean}
	case g.inContent:
		return &DeltaMessage{Content: &clean}
	case !g.sawAnyChannel:
		return &DeltaMessage{Content: &clean}
	default:
		return &DeltaMessage{Reasoning: &clean}
	}
}

// FinalizeStreaming is a no-op for Gemma 4.
func (g *Gemma4) FinalizeStreaming(string) *DeltaMessage { return nil }

var gemmaMarkers = []string{
	"<|channel>", "<channel|>", "<|turn>", "<turn|>",
	"thought\n", "content\n", "final\n",
}

func gemmaStripMarkers(s string) string {
	for _, m := range gemmaMarkers {
		s = strings.ReplaceAll(s, m, "")
	}
	return s
}
