// SPDX-License-Identifier: Apache-2.0

package toolparsers

// The registry maps the parser names a request may ask for to a constructor.
// Names mirror the reference parser manager so an existing client config keeps
// working unchanged. Each lookup returns a fresh parser with its streaming
// state reset, because a parser instance is stateful across the deltas of one
// response and must not be shared between requests.

type constructor func() ToolParser

func newReset(p ToolParser) ToolParser {
	p.Reset()
	return p
}

var registry = map[string]constructor{
	"llama":            func() ToolParser { return newReset(&LlamaParser{}) },
	"llama3":           func() ToolParser { return newReset(&LlamaParser{}) },
	"llama4":           func() ToolParser { return newReset(&LlamaParser{}) },
	"granite":          func() ToolParser { return newReset(&GraniteParser{}) },
	"granite3":         func() ToolParser { return newReset(&GraniteParser{}) },
	"kimi":             func() ToolParser { return newReset(&KimiParser{}) },
	"kimi_k2":          func() ToolParser { return newReset(&KimiParser{}) },
	"moonshot":         func() ToolParser { return newReset(&KimiParser{}) },
	"nemotron":         func() ToolParser { return newReset(&NemotronParser{}) },
	"nemotron3":        func() ToolParser { return newReset(&NemotronParser{}) },
	"deepseek":         func() ToolParser { return newReset(&DeepSeekParser{}) },
	"deepseek_v3":      func() ToolParser { return newReset(&DeepSeekParser{}) },
	"deepseek_r1":      func() ToolParser { return newReset(&DeepSeekParser{}) },
	"deepseek_v31":     func() ToolParser { return newReset(&DeepSeekV31Parser{}) },
	"deepseek_r1_0528": func() ToolParser { return newReset(&DeepSeekV31Parser{}) },
	"xlam":             func() ToolParser { return newReset(&XLAMParser{}) },
	"qwen":             func() ToolParser { return newReset(&QwenParser{}) },
	"qwen3":            func() ToolParser { return newReset(&QwenParser{}) },
	"qwen3_xml":        func() ToolParser { return newReset(&QwenParser{}) },
	"qwen3_coder_xml":  func() ToolParser { return newReset(&Qwen3CoderParser{}) },
	"functionary":      func() ToolParser { return newReset(&FunctionaryParser{}) },
	"meetkai":          func() ToolParser { return newReset(&FunctionaryParser{}) },
	"glm47":            func() ToolParser { return newReset(&Glm47Parser{}) },
	"glm4":             func() ToolParser { return newReset(&Glm47Parser{}) },
	"gemma4":           func() ToolParser { return newReset(&Gemma4Parser{}) },
	"gemma_4":          func() ToolParser { return newReset(&Gemma4Parser{}) },
	"minimax":          func() ToolParser { return newReset(&MiniMaxParser{}) },
	"minimax_m2":       func() ToolParser { return newReset(&MiniMaxParser{}) },
	"mistral":          func() ToolParser { return newReset(&MistralParser{}) },
	"hermes":           func() ToolParser { return newReset(&HermesParser{}) },
	"nous":             func() ToolParser { return newReset(&HermesParser{}) },
	"qwen3_coder":      func() ToolParser { return newReset(&HermesParser{}) },
	"harmony":          func() ToolParser { return newReset(&HarmonyParser{}) },
	"gpt-oss":          func() ToolParser { return newReset(&HarmonyParser{}) },
	"seed_oss":         func() ToolParser { return newReset(&SeedOssParser{}) },
	"seed":             func() ToolParser { return newReset(&SeedOssParser{}) },
	"gpt_oss":          func() ToolParser { return newReset(&SeedOssParser{}) },
	"auto":             func() ToolParser { return newReset(&AutoParser{}) },
	"generic":          func() ToolParser { return newReset(&AutoParser{}) },
}

// Get returns a fresh parser for the named format. The boolean is false when
// the name is unknown.
func Get(name string) (ToolParser, bool) {
	c, ok := registry[name]
	if !ok {
		return nil, false
	}
	return c(), true
}

// Names returns every registered parser name. Order is unspecified.
func Names() []string {
	out := make([]string, 0, len(registry))
	for name := range registry {
		out = append(out, name)
	}
	return out
}
