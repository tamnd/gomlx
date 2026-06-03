// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"sync"
	"testing"
)

func toolNames(defs []map[string]any) []string {
	names := make([]string, 0, len(defs))
	for _, d := range defs {
		fn := d["function"].(map[string]any)
		names = append(names, fn["name"].(string))
	}
	return names
}

func TestRegistrySetAndListSorted(t *testing.T) {
	r := NewRegistry(Config{})
	r.SetTools("zeta", []Tool{{Name: "a"}})
	r.SetTools("alpha", []Tool{{Name: "b"}})

	tools := r.Tools()
	if len(tools) != 2 {
		t.Fatalf("want 2 tools, got %d", len(tools))
	}
	// Sorted by full name regardless of insertion order.
	if tools[0].FullName() != "alpha__b" || tools[1].FullName() != "zeta__a" {
		t.Fatalf("not sorted: %q %q", tools[0].FullName(), tools[1].FullName())
	}
	// SetTools stamps the server name onto each tool.
	if tools[0].ServerName != "alpha" {
		t.Fatalf("server name not stamped: %q", tools[0].ServerName)
	}
}

func TestRegistryRemoveServer(t *testing.T) {
	r := NewRegistry(Config{})
	r.SetTools("s", []Tool{{Name: "a"}})
	r.RemoveServer("s")
	if got := r.Tools(); len(got) != 0 {
		t.Fatalf("tools should be gone after remove, got %v", got)
	}
}

func TestRegistrySetToolsReplaces(t *testing.T) {
	r := NewRegistry(Config{})
	r.SetTools("s", []Tool{{Name: "old"}})
	r.SetTools("s", []Tool{{Name: "new"}})
	tools := r.Tools()
	if len(tools) != 1 || tools[0].Name != "new" {
		t.Fatalf("second SetTools should replace, got %v", tools)
	}
}

func TestRegistryHidesHighRiskFromList(t *testing.T) {
	r := NewRegistry(Config{})
	r.SetTools("s", []Tool{
		{Name: "read"},
		{Name: "exec"}, // high risk
	})
	names := toolNames(r.OpenAITools())
	if len(names) != 1 || names[0] != "s__read" {
		t.Fatalf("high-risk tool should be hidden, got %v", names)
	}
}

func TestRegistryAllowlistRevealsHighRisk(t *testing.T) {
	r := NewRegistry(Config{AllowedHighRiskTools: []string{"s__exec"}})
	r.SetTools("s", []Tool{{Name: "read"}, {Name: "exec"}})
	names := toolNames(r.OpenAITools())
	if len(names) != 2 {
		t.Fatalf("allowlisted high-risk tool should be listed, got %v", names)
	}
}

func TestRegistrySkipValidationListsEverything(t *testing.T) {
	cfg := Config{Servers: map[string]ServerConfig{
		"trusted": {Name: "trusted", SkipSecurityValidation: true},
	}}
	r := NewRegistry(cfg)
	r.SetTools("trusted", []Tool{{Name: "exec"}})
	if names := toolNames(r.OpenAITools()); len(names) != 1 {
		t.Fatalf("a trusted server's high-risk tool should be listed, got %v", names)
	}
}

func TestRegistryResolveAppliesGate(t *testing.T) {
	r := NewRegistry(Config{})
	r.SetTools("s", []Tool{
		{Name: "read"},
		{Name: "exec"},
	})

	// A safe tool with clean args resolves.
	if _, err := r.Resolve("s__read", map[string]any{"path": "data/x.txt"}); err != nil {
		t.Fatalf("safe call should resolve: %v", err)
	}
	// A high-risk tool is refused.
	if _, err := r.Resolve("s__exec", nil); err == nil {
		t.Fatal("high-risk tool should be refused at resolve")
	}
	// A safe tool with a dangerous argument is refused.
	if _, err := r.Resolve("s__read", map[string]any{"path": "../../etc/passwd"}); err == nil {
		t.Fatal("dangerous argument should be refused at resolve")
	}
	// An unknown tool is an error.
	if _, err := r.Resolve("s__missing", nil); err == nil {
		t.Fatal("unknown tool should error")
	}
}

func TestRegistryResolveSkipBypassesGate(t *testing.T) {
	cfg := Config{Servers: map[string]ServerConfig{
		"trusted": {Name: "trusted", SkipSecurityValidation: true},
	}}
	r := NewRegistry(cfg)
	r.SetTools("trusted", []Tool{{Name: "exec"}})
	// Both the high-risk name and the dangerous argument are allowed for a
	// trusted server.
	if _, err := r.Resolve("trusted__exec", map[string]any{"cmd": "/etc/init.d/x"}); err != nil {
		t.Fatalf("trusted server should bypass the gate: %v", err)
	}
}

func TestRegistryConcurrentAccess(t *testing.T) {
	r := NewRegistry(Config{})
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.SetTools("s", []Tool{{Name: "read"}})
			_ = r.OpenAITools()
			_, _ = r.Resolve("s__read", nil)
			_ = r.Tools()
		}()
	}
	wg.Wait()
}
