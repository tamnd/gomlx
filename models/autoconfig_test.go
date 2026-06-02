// SPDX-License-Identifier: Apache-2.0

package models

import "testing"

func TestDetectModelConfigPatterns(t *testing.T) {
	cases := []struct {
		name string
		path string
		tool string
		reas string
		hybr bool
		spec bool
	}{
		{"deepseek-v4", "mlx-community/DeepSeek-V4-Flash", "deepseek", "", false, true},
		{"deepseek-r1-0528", "deepseek-ai/DeepSeek-R1-0528", "deepseek_v31", "deepseek_r1", false, true},
		{"deepseek-r1", "deepseek-ai/DeepSeek-R1", "deepseek", "deepseek_r1", false, true},
		{"qwen3.5 hybrid", "mlx-community/Qwen3.5-4B", "hermes", "qwen3", true, false},
		{"qwen3.6", "org/Qwen3.6-X", "qwen3_coder_xml", "qwen3", true, false},
		{"qwen3 plain", "Qwen/Qwen3-8B", "hermes", "qwen3", false, true},
		{"gpt-oss", "openai/gpt-oss-20b", "harmony", "harmony", false, true},
		{"gemma4", "google/gemma-4-9b", "gemma4", "gemma4", false, true},
		{"llama", "meta-llama/Llama-3.1-8B", "llama", "", false, true},
		{"mamba", "state-spaces/mamba-2.8b", "", "", true, false},
		{"granite4", "ibm/granite-4-tiny", "hermes", "", true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := DetectModelConfig(c.path)
			if cfg.ToolCallParser != c.tool {
				t.Errorf("tool: got %q want %q", cfg.ToolCallParser, c.tool)
			}
			if cfg.ReasoningParser != c.reas {
				t.Errorf("reasoning: got %q want %q", cfg.ReasoningParser, c.reas)
			}
			if cfg.IsHybrid != c.hybr {
				t.Errorf("hybrid: got %v want %v", cfg.IsHybrid, c.hybr)
			}
			if cfg.SupportsSpecDecode != c.spec {
				t.Errorf("spec: got %v want %v", cfg.SupportsSpecDecode, c.spec)
			}
		})
	}
}

func TestDetectModelConfigSpecificityOrder(t *testing.T) {
	// "deepseek r1 0528" must match the v3.1/r1-0528 entry before the
	// generic "deepseek.*r1" entry.
	cfg := DetectModelConfig("DeepSeek-R1-0528-Qwen3")
	if cfg.ToolCallParser != "deepseek_v31" {
		t.Errorf("specificity order broken: got %q", cfg.ToolCallParser)
	}
}

func TestDetectModelConfigUnknownDefault(t *testing.T) {
	cfg := DetectModelConfig("totally-unknown-arch-xyz")
	if cfg.ToolCallParser != "" || !cfg.SupportsSpecDecode || cfg.IsHybrid {
		t.Errorf("unknown should yield permissive default, got %+v", cfg)
	}
}
