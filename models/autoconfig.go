// SPDX-License-Identifier: Apache-2.0

package models

import "regexp"

// ModelConfig is the auto-detected configuration for a model family. It carries
// parser defaults plus capability gates. Defaults err on the side of
// "supported"; known-incompatible families set the flag explicitly. Ported from
// model_auto_config.py.
type ModelConfig struct {
	// Parser defaults.
	ToolCallParser   string
	ReasoningParser  string
	DefaultMaxTokens int // 0 means unset

	// Architecture / capability gates.
	// IsHybrid means the model uses linear-attention or recurrent layers
	// (GatedDeltaNet, Mamba, Jamba, ...); such models disable optimizations
	// that rely on chunked-batched forward.
	IsHybrid bool
	// SupportsSpecDecode controls suffix/draft-model speculative decoding.
	// Disabled for hybrid models because the batched-verify path derails
	// generation. Pure-attention models are safe (default true).
	SupportsSpecDecode bool

	// SuffixDecodingTier is one of: unknown, agent, structured, neutral, avoid.
	SuffixDecodingTier string
	SuffixBenchSpeedup map[string]float64
}

func defaultModelConfig() ModelConfig {
	return ModelConfig{
		SupportsSpecDecode: true,
		SuffixDecodingTier: "unknown",
	}
}

type pattern struct {
	re  *regexp.Regexp
	cfg ModelConfig
}

// modelPatterns is ordered: first match wins, most specific first. Mirrors
// _MODEL_PATTERNS in model_auto_config.py.
var modelPatterns []pattern

func init() {
	def := defaultModelConfig()
	add := func(expr string, cfg ModelConfig) {
		modelPatterns = append(modelPatterns, pattern{
			re:  regexp.MustCompile("(?i)" + expr),
			cfg: cfg,
		})
	}

	add(`deepseek.*v4`, merge(def, ModelConfig{ToolCallParser: "deepseek"}))
	add(`deepseek.*(v3\.1|r1[-_]?0528)`, merge(def, ModelConfig{ToolCallParser: "deepseek_v31", ReasoningParser: "deepseek_r1"}))
	add(`deepseek.*r1`, merge(def, ModelConfig{ToolCallParser: "deepseek", ReasoningParser: "deepseek_r1"}))
	add(`deepseek`, merge(def, ModelConfig{ToolCallParser: "deepseek"}))
	add(`qwopus`, merge(def, ModelConfig{ToolCallParser: "hermes", ReasoningParser: "qwen3", IsHybrid: true, SupportsSpecDecode: false}))
	add(`qwen3[-_]?(coder[-_]?next|next)`, merge(def, ModelConfig{ToolCallParser: "hermes", IsHybrid: true, SupportsSpecDecode: false}))
	add(`qwen3\.6`, merge(def, ModelConfig{ToolCallParser: "qwen3_coder_xml", ReasoningParser: "qwen3", IsHybrid: true, SupportsSpecDecode: false}))
	add(`qwen3\.5`, merge(def, ModelConfig{ToolCallParser: "hermes", ReasoningParser: "qwen3", IsHybrid: true, SupportsSpecDecode: false}))
	add(`qwen3[-_]?coder`, merge(def, ModelConfig{ToolCallParser: "hermes"}))
	add(`qwen3`, merge(def, ModelConfig{ToolCallParser: "hermes", ReasoningParser: "qwen3"}))
	add(`glm[-_]?4`, merge(def, ModelConfig{ToolCallParser: "glm47"}))
	add(`minimax`, merge(def, ModelConfig{ToolCallParser: "minimax", ReasoningParser: "minimax"}))
	add(`gpt[-_]?oss`, merge(def, ModelConfig{ToolCallParser: "harmony", ReasoningParser: "harmony"}))
	add(`kimi`, merge(def, ModelConfig{ToolCallParser: "kimi"}))
	add(`magistral`, merge(def, ModelConfig{ToolCallParser: "hermes", ReasoningParser: "qwen3"}))
	add(`mistral|devstral`, merge(def, ModelConfig{ToolCallParser: "hermes"}))
	add(`gemma[-_]?4`, merge(def, ModelConfig{ToolCallParser: "gemma4", ReasoningParser: "gemma4"}))
	add(`gemma`, merge(def, ModelConfig{ToolCallParser: "hermes"}))
	add(`hermes`, merge(def, ModelConfig{ToolCallParser: "hermes"}))
	add(`llama`, merge(def, ModelConfig{ToolCallParser: "llama"}))
	add(`phi[-_]?[34]`, merge(def, ModelConfig{ToolCallParser: "hermes"}))
	add(`granite[-_]?4`, merge(def, ModelConfig{ToolCallParser: "hermes", IsHybrid: true, SupportsSpecDecode: false}))
	add(`smollm3`, merge(def, ModelConfig{ToolCallParser: "hermes", ReasoningParser: "qwen3"}))
	add(`mamba|jamba|rwkv`, merge(def, ModelConfig{IsHybrid: true, SupportsSpecDecode: false}))
}

// merge overlays the non-zero parser fields of override onto base. A hybrid
// entry is marked hybrid and has speculative decoding disabled (matching every
// hybrid family in the reference table); non-hybrid entries keep the base
// default of SupportsSpecDecode=true. Parser defaults always overlay.
func merge(base, override ModelConfig) ModelConfig {
	out := base
	if override.ToolCallParser != "" {
		out.ToolCallParser = override.ToolCallParser
	}
	if override.ReasoningParser != "" {
		out.ReasoningParser = override.ReasoningParser
	}
	if override.DefaultMaxTokens != 0 {
		out.DefaultMaxTokens = override.DefaultMaxTokens
	}
	if override.IsHybrid {
		out.IsHybrid = true
		out.SupportsSpecDecode = false
	}
	if override.SuffixDecodingTier != "" {
		out.SuffixDecodingTier = override.SuffixDecodingTier
	}
	if override.SuffixBenchSpeedup != nil {
		out.SuffixBenchSpeedup = override.SuffixBenchSpeedup
	}
	return out
}

// DetectModelConfig infers a ModelConfig from a model name or path. It consults
// the alias profile first, then falls back to the ordered regex patterns. If
// nothing matches it returns the permissive default.
func DetectModelConfig(modelPath string) ModelConfig {
	cfg := defaultModelConfig()

	// Alias profile takes precedence when present.
	if p, ok := ResolveProfile(modelPath); ok {
		if p.ToolCallParser != "" {
			cfg.ToolCallParser = p.ToolCallParser
		}
		if p.ReasoningParser != "" {
			cfg.ReasoningParser = p.ReasoningParser
		}
		cfg.IsHybrid = p.IsHybrid
		cfg.SupportsSpecDecode = p.SupportsSpecDecode
		if p.SuffixDecodingTier != "" {
			cfg.SuffixDecodingTier = p.SuffixDecodingTier
		}
		if p.SuffixBenchSpeedup != nil {
			cfg.SuffixBenchSpeedup = p.SuffixBenchSpeedup
		}
		// An alias profile that names no parser still falls through to the
		// regex below to fill parser defaults from the hf_path.
		if cfg.ToolCallParser != "" || cfg.ReasoningParser != "" {
			return cfg
		}
		modelPath = p.HFPath
	}

	name := normalizeName(modelPath)
	for _, pat := range modelPatterns {
		if pat.re.MatchString(name) {
			return pat.cfg
		}
	}
	return cfg
}
