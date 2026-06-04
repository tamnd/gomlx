// SPDX-License-Identifier: Apache-2.0

package tokenizer

import "strings"

// Each model family wraps a conversation in its own marker format before the
// text is tokenized, and a model trained on one format produces garbage when fed
// another. ApplyChatTemplate renders a conversation for a given family: the Qwen
// families use ChatML, Llama uses the Llama 3 header format, Mistral uses the
// [INST] format, and Gemma uses its <start_of_turn> format. An unrecognized
// family falls back to ChatML, which is the most common modern default.
func ApplyChatTemplate(arch string, msgs []ChatMsg, addGenerationPrompt bool) string {
	switch arch {
	case "llama":
		return applyLlama3(msgs, addGenerationPrompt)
	case "mistral":
		return applyMistral(msgs)
	case "gemma":
		return applyGemma(msgs, addGenerationPrompt)
	default: // qwen3, qwen2, and any unknown family
		return ApplyChatML(msgs, addGenerationPrompt)
	}
}

// applyLlama3 renders messages in the Llama 3 format: a single begin-of-text
// marker, then each turn as a role header block ended by <|eot_id|>. Content is
// trimmed, matching the reference template. With addGenerationPrompt set the
// string ends with an open assistant header so the model continues from there.
func applyLlama3(msgs []ChatMsg, addGenerationPrompt bool) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|>")
	for _, m := range msgs {
		b.WriteString("<|start_header_id|>")
		b.WriteString(m.Role)
		b.WriteString("<|end_header_id|>\n\n")
		b.WriteString(strings.TrimSpace(m.Content))
		b.WriteString("<|eot_id|>")
	}
	if addGenerationPrompt {
		b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	}
	return b.String()
}

// applyMistral renders messages in the classic Mistral instruct format: a single
// begin-of-sequence marker, each user turn wrapped in [INST] ... [/INST], and
// each assistant turn followed by an end-of-sequence marker. Mistral has no
// system markers, so a leading system message is folded into the first user turn.
// The prompt ends after the final [/INST], which is where the model continues, so
// there is no separate generation-prompt marker to add.
func applyMistral(msgs []ChatMsg) string {
	var b strings.Builder
	b.WriteString("<s>")
	var system string
	for _, m := range msgs {
		switch m.Role {
		case "system":
			if system != "" {
				system += "\n\n"
			}
			system += m.Content
		case "user":
			b.WriteString("[INST] ")
			if system != "" {
				b.WriteString(system)
				b.WriteString("\n\n")
				system = ""
			}
			b.WriteString(m.Content)
			b.WriteString(" [/INST]")
		case "assistant":
			b.WriteString(m.Content)
			b.WriteString("</s>")
		}
	}
	return b.String()
}

// applyGemma renders messages in Gemma's format: a single begin-of-sequence
// marker, then each turn wrapped as <start_of_turn>role\n ... <end_of_turn>\n
// with the content trimmed. Gemma names the assistant role "model" and has no
// system role, so a leading system message is folded into the first user turn.
// With addGenerationPrompt set the string ends with an open model turn so the
// model continues from there.
func applyGemma(msgs []ChatMsg, addGenerationPrompt bool) string {
	var b strings.Builder
	b.WriteString("<bos>")
	var system string
	for _, m := range msgs {
		role, content := m.Role, m.Content
		switch role {
		case "system":
			if system != "" {
				system += "\n\n"
			}
			system += content
			continue
		case "assistant":
			role = "model"
		}
		if role == "user" && system != "" {
			content = system + "\n\n" + content
			system = ""
		}
		b.WriteString("<start_of_turn>")
		b.WriteString(role)
		b.WriteString("\n")
		b.WriteString(strings.TrimSpace(content))
		b.WriteString("<end_of_turn>\n")
	}
	if addGenerationPrompt {
		b.WriteString("<start_of_turn>model\n")
	}
	return b.String()
}
