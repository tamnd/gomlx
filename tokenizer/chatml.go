// SPDX-License-Identifier: Apache-2.0

package tokenizer

import "strings"

// ChatMsg is one conversation turn for the chat template. It is a minimal local
// type so this package stays free of imports from the engine or api layers.
type ChatMsg struct {
	Role    string
	Content string
}

// ApplyChatML renders messages in the ChatML format Qwen3 uses. Each turn is
// wrapped in <|im_start|>role ... <|im_end|> markers. When addGenerationPrompt
// is set, the string ends with an open assistant turn so the model continues
// from there. Thinking mode is left to the model: the prompt does not inject an
// empty <think> block, matching the default Qwen3 template.
func ApplyChatML(msgs []ChatMsg, addGenerationPrompt bool) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteByte('\n')
		b.WriteString(m.Content)
		b.WriteString("<|im_end|>\n")
	}
	if addGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
	}
	return b.String()
}
