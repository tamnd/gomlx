// SPDX-License-Identifier: Apache-2.0

package models

import "testing"

func TestAliasesLoaded(t *testing.T) {
	if len(Aliases()) == 0 {
		t.Fatal("no aliases loaded from embedded aliases.json")
	}
	// The reference file has 66 aliases at v0.6.71; allow growth but not zero.
	if got := len(Aliases()); got < 60 {
		t.Errorf("expected >=60 aliases, got %d", got)
	}
}

func TestResolveModelKnownAlias(t *testing.T) {
	p, ok := ResolveProfile("qwen3.5-4b")
	if !ok {
		t.Fatal("qwen3.5-4b should resolve")
	}
	if p.HFPath != "mlx-community/Qwen3.5-4B-MLX-4bit" {
		t.Errorf("unexpected hf_path: %s", p.HFPath)
	}
	if !p.IsHybrid {
		t.Error("qwen3.5-4b should be hybrid")
	}
	if got := ResolveModel("qwen3.5-4b"); got != p.HFPath {
		t.Errorf("ResolveModel mismatch: %s", got)
	}
}

func TestResolveModelUnknownPassthrough(t *testing.T) {
	const raw = "some-org/Some-Model-MLX"
	if got := ResolveModel(raw); got != raw {
		t.Errorf("unknown name should pass through unchanged, got %s", got)
	}
	if _, ok := ResolveProfile(raw); ok {
		t.Error("unknown name should not resolve a profile")
	}
}

func TestReverseIndexDeterministic(t *testing.T) {
	// hf_path -> alias must be stable across calls (sorted preference).
	p, _ := ResolveProfile("qwen3.5-4b")
	a1 := AliasForPath(p.HFPath)
	a2 := AliasForPath(p.HFPath)
	if a1 != a2 || a1 == "" {
		t.Errorf("reverse index not deterministic: %q vs %q", a1, a2)
	}
	// Resolving by path should round-trip to a profile.
	if _, ok := ResolveProfile(p.HFPath); !ok {
		t.Error("resolving by hf_path should succeed")
	}
}
