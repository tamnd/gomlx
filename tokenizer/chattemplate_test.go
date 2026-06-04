// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"strings"
	"testing"
)

var convo = []ChatMsg{
	{Role: "system", Content: "Be brief."},
	{Role: "user", Content: "Hi"},
}

func TestApplyChatTemplateLlama3(t *testing.T) {
	got := ApplyChatTemplate("llama", convo, true)
	want := "<|begin_of_text|>" +
		"<|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>" +
		"<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\n"
	if got != want {
		t.Errorf("llama3 template:\n got %q\nwant %q", got, want)
	}
}

func TestApplyChatTemplateLlama3NoGenPrompt(t *testing.T) {
	got := ApplyChatTemplate("llama", []ChatMsg{{Role: "user", Content: "Hi"}}, false)
	if strings.Contains(got, "assistant") {
		t.Errorf("without a generation prompt there should be no open assistant header: %q", got)
	}
	if !strings.HasSuffix(got, "<|eot_id|>") {
		t.Errorf("turn should end with eot: %q", got)
	}
}

func TestApplyChatTemplateMistral(t *testing.T) {
	got := ApplyChatTemplate("mistral", convo, true)
	// System folds into the first user turn; the prompt ends ready for the model.
	want := "<s>[INST] Be brief.\n\nHi [/INST]"
	if got != want {
		t.Errorf("mistral template:\n got %q\nwant %q", got, want)
	}
}

func TestApplyChatTemplateMistralMultiTurn(t *testing.T) {
	msgs := []ChatMsg{
		{Role: "user", Content: "Hi"},
		{Role: "assistant", Content: "Hello"},
		{Role: "user", Content: "Bye"},
	}
	got := ApplyChatTemplate("mistral", msgs, true)
	want := "<s>[INST] Hi [/INST]Hello</s>[INST] Bye [/INST]"
	if got != want {
		t.Errorf("mistral multi-turn:\n got %q\nwant %q", got, want)
	}
}

func TestApplyChatTemplateChatMLForQwen(t *testing.T) {
	for _, arch := range []string{"qwen3", "qwen2", "somethingelse"} {
		got := ApplyChatTemplate(arch, convo, true)
		if !strings.HasPrefix(got, "<|im_start|>system\nBe brief.<|im_end|>\n") {
			t.Errorf("arch %q should use ChatML, got %q", arch, got)
		}
		if !strings.HasSuffix(got, "<|im_start|>assistant\n") {
			t.Errorf("arch %q ChatML should end with an open assistant turn, got %q", arch, got)
		}
	}
}

func TestApplyChatTemplateMatchesChatMLForUnknown(t *testing.T) {
	// The fallback must be byte-identical to ApplyChatML so the default path is
	// unchanged for families that were already correct.
	if ApplyChatTemplate("", convo, true) != ApplyChatML(convo, true) {
		t.Error("unknown arch must fall back to exactly ApplyChatML")
	}
}
