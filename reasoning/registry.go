// SPDX-License-Identifier: Apache-2.0

package reasoning

import (
	"fmt"
	"sort"
)

// constructors maps a parser name to its factory. New instances are returned
// per request because the streaming parsers are stateful.
var constructors = map[string]func() ReasoningParser{
	"gemma4":      func() ReasoningParser { return NewGemma4() },
	"qwen3":       func() ReasoningParser { return NewQwen3() },
	"deepseek_r1": func() ReasoningParser { return NewDeepSeekR1() },
	"glm4":        func() ReasoningParser { return NewGLM4() },
	"gpt_oss":     func() ReasoningParser { return NewGptOss() },
	"harmony":     func() ReasoningParser { return NewHarmony() },
	"minimax":     func() ReasoningParser { return NewMiniMax() },
}

// Get returns a fresh parser instance for the named family.
func Get(name string) (ReasoningParser, error) {
	ctor, ok := constructors[name]
	if !ok {
		return nil, fmt.Errorf("reasoning parser %q not found; available: %v", name, List())
	}
	return ctor(), nil
}

// List returns the registered parser names, sorted.
func List() []string {
	names := make([]string, 0, len(constructors))
	for name := range constructors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
