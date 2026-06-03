// SPDX-License-Identifier: Apache-2.0

package agents

import (
	"slices"
	"strings"
	"testing"
)

func TestAllProfilesLoad(t *testing.T) {
	names := Names()
	want := []string{
		"aider", "cline", "codex", "generic", "goose", "hermes",
		"langchain", "openclaude", "opencode", "openhands", "pydanticai", "smolagents",
	}
	if !slices.Equal(names, want) {
		t.Fatalf("profile names = %v, want %v", names, want)
	}
	// Every profile keeps its own name and a non-empty display name.
	for _, n := range names {
		p, ok := Get(n)
		if !ok {
			t.Fatalf("Get(%q) missing", n)
		}
		if p.Name != n {
			t.Errorf("profile %q has Name %q", n, p.Name)
		}
		if p.DisplayName == "" {
			t.Errorf("profile %q has empty display name", n)
		}
	}
}

func TestGetUnknownAndGeneric(t *testing.T) {
	if _, ok := Get("does-not-exist"); ok {
		t.Fatal("Get of an unknown profile should report not found")
	}
	p, ok := GetOrGeneric("does-not-exist")
	if !ok || p.Name != "generic" {
		t.Fatalf("GetOrGeneric should fall back to generic, got %q ok=%v", p.Name, ok)
	}
}

func TestCapabilityDefaults(t *testing.T) {
	// generic.yaml declares function_calling and streaming true, vision and
	// reasoning false, which is also the default when a flag is omitted.
	g, _ := Get("generic")
	if !g.NeedsFunctionCalling || !g.NeedsStreaming {
		t.Errorf("generic should need function calling and streaming: %+v", g)
	}
	if g.NeedsVision || g.NeedsReasoning {
		t.Errorf("generic should not need vision or reasoning: %+v", g)
	}
	// hermes works with thinking models.
	if h, _ := Get("hermes"); !h.NeedsReasoning {
		t.Error("hermes should need reasoning")
	}
}

func TestRenderEnvConfig(t *testing.T) {
	g, _ := Get("generic")
	rc := g.RenderConfig("http://localhost:8000/v1", "qwen3.5-4b", "")
	if rc.Type != "env" {
		t.Fatalf("generic config type = %q, want env", rc.Type)
	}
	if rc.EnvVars["OPENAI_BASE_URL"] != "http://localhost:8000/v1" {
		t.Errorf("base url not substituted: %q", rc.EnvVars["OPENAI_BASE_URL"])
	}
	if rc.EnvVars["OPENAI_MODEL"] != "qwen3.5-4b" {
		t.Errorf("model not substituted: %q", rc.EnvVars["OPENAI_MODEL"])
	}
}

func TestRenderFileConfigSubstitutes(t *testing.T) {
	// hermes uses a yaml template; the rendered body carries the base url and
	// model and no leftover placeholders.
	h, _ := Get("hermes")
	rc := h.RenderConfig("http://localhost:8000/v1", "qwen3.5-9b", "")
	if rc.Type != "yaml" {
		t.Fatalf("hermes config type = %q, want yaml", rc.Type)
	}
	if !strings.Contains(rc.Content, "http://localhost:8000/v1") {
		t.Errorf("base url missing from rendered config:\n%s", rc.Content)
	}
	if !strings.Contains(rc.Content, "qwen3.5-9b") {
		t.Errorf("model missing from rendered config:\n%s", rc.Content)
	}
	if strings.Contains(rc.Content, "{base_url}") || strings.Contains(rc.Content, "{model_id}") {
		t.Errorf("placeholders left unrendered:\n%s", rc.Content)
	}
}

func TestRenderBaseURLNoV1(t *testing.T) {
	p := Profile{Config: ConfigSpec{
		Type:    "env",
		EnvVars: map[string]string{"FULL": "{base_url}", "BARE": "{base_url_no_v1}"},
	}}
	rc := p.RenderConfig("http://host:9000/v1", "m", "")
	if rc.EnvVars["FULL"] != "http://host:9000/v1" {
		t.Errorf("FULL = %q", rc.EnvVars["FULL"])
	}
	if rc.EnvVars["BARE"] != "http://host:9000" {
		t.Errorf("BARE = %q, want the /v1 stripped", rc.EnvVars["BARE"])
	}
}

func TestExtraToolTagsParsed(t *testing.T) {
	// hermes carries one extra tool tag pair, ["[Calling tool", "\n"].
	h, _ := Get("hermes")
	tags := h.Streaming.ExtraToolTags
	if len(tags) != 1 || tags[0].Open != "[Calling tool" || tags[0].Close != "\n" {
		t.Fatalf("hermes extra tool tags = %+v", tags)
	}
	if h.Streaming.MaxTools != 62 {
		t.Errorf("hermes max_tools = %d, want 62", h.Streaming.MaxTools)
	}
}

func TestQueryTimeoutDefault(t *testing.T) {
	// A profile whose testing section omits a timeout gets the default.
	c, _ := Get("cline")
	if c.Testing.QueryTimeout != defaultQueryTimeout {
		t.Errorf("cline query timeout = %d, want default %d", c.Testing.QueryTimeout, defaultQueryTimeout)
	}
}

func TestListSortedByStarsDescending(t *testing.T) {
	list := List()
	if len(list) != len(Names()) {
		t.Fatalf("List len %d != Names len %d", len(list), len(Names()))
	}
	for i := 1; i < len(list); i++ {
		if list[i-1].Stars < list[i].Stars {
			t.Fatalf("list not sorted by stars descending at %d: %d < %d",
				i, list[i-1].Stars, list[i].Stars)
		}
	}
}
