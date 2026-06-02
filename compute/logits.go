// SPDX-License-Identifier: Apache-2.0

package compute

// LogitsProcessor adjusts a logits row in place before sampling, using the
// tokens generated so far. It covers the three penalties the serving layer
// exposes: a repetition penalty over a recent context window and the OpenAI
// presence and frequency penalties over the whole history. All three are
// disabled at their identity values (RepetitionPenalty 1, the others 0), so a
// zero-value LogitsProcessor is a no-op.
type LogitsProcessor struct {
	// RepetitionPenalty divides positive logits and multiplies negative logits
	// for tokens seen in the recent window. 1.0 disables it.
	RepetitionPenalty float64
	// RepetitionContextSize bounds how many of the most recent tokens the
	// repetition penalty considers. 0 means the whole history.
	RepetitionContextSize int
	// PresencePenalty is subtracted once from the logit of any token that has
	// appeared at least once.
	PresencePenalty float64
	// FrequencyPenalty is subtracted from a token's logit once per occurrence.
	FrequencyPenalty float64
}

// Enabled reports whether any penalty would change the logits.
func (p LogitsProcessor) Enabled() bool {
	return (p.RepetitionPenalty != 0 && p.RepetitionPenalty != 1) ||
		p.PresencePenalty != 0 || p.FrequencyPenalty != 0
}

// Apply mutates logits in place given the tokens generated so far. tokens holds
// the full generated history (and may include the prompt); ids outside the
// logits range are ignored.
func (p LogitsProcessor) Apply(logits []float32, tokens []int) {
	if len(logits) == 0 || len(tokens) == 0 {
		return
	}

	// Repetition penalty over the recent window. Matches the reference: a
	// positive logit is divided by the penalty, a negative logit multiplied,
	// which pushes both toward zero.
	if p.RepetitionPenalty != 0 && p.RepetitionPenalty != 1 {
		window := tokens
		if p.RepetitionContextSize > 0 && len(window) > p.RepetitionContextSize {
			window = window[len(window)-p.RepetitionContextSize:]
		}
		seen := make(map[int]struct{}, len(window))
		for _, tok := range window {
			if tok < 0 || tok >= len(logits) {
				continue
			}
			if _, dup := seen[tok]; dup {
				continue
			}
			seen[tok] = struct{}{}
			if logits[tok] > 0 {
				logits[tok] = float32(float64(logits[tok]) / p.RepetitionPenalty)
			} else {
				logits[tok] = float32(float64(logits[tok]) * p.RepetitionPenalty)
			}
		}
	}

	// Presence and frequency penalties over the full history. Both are additive
	// in logit space: logit -= frequency*count + presence*(count > 0).
	if p.PresencePenalty != 0 || p.FrequencyPenalty != 0 {
		counts := make(map[int]int, len(tokens))
		for _, tok := range tokens {
			if tok < 0 || tok >= len(logits) {
				continue
			}
			counts[tok]++
		}
		for tok, c := range counts {
			adj := p.FrequencyPenalty*float64(c) + p.PresencePenalty
			logits[tok] = float32(float64(logits[tok]) - adj)
		}
	}
}
