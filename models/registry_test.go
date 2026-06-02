// SPDX-License-Identifier: Apache-2.0

package models

import "testing"

func TestOwnershipRegistry(t *testing.T) {
	r := NewOwnershipRegistry()
	const path = "mlx-community/Qwen3.5-4B-MLX-4bit"

	if err := r.Acquire(path, "engine-a"); err != nil {
		t.Fatalf("first acquire should succeed: %v", err)
	}
	// Re-acquire by the same owner is idempotent.
	if err := r.Acquire(path, "engine-a"); err != nil {
		t.Fatalf("same-owner re-acquire should succeed: %v", err)
	}
	// A different owner is rejected.
	if err := r.Acquire(path, "engine-b"); err == nil {
		t.Fatal("conflicting acquire should fail")
	}
	if got := r.Owner(path); got != "engine-a" {
		t.Errorf("owner: got %q want engine-a", got)
	}
	// Release by a non-owner is a no-op.
	r.Release(path, "engine-b")
	if r.Owner(path) != "engine-a" {
		t.Error("non-owner release must not drop ownership")
	}
	// Release by the owner frees it.
	r.Release(path, "engine-a")
	if r.Owner(path) != "" {
		t.Error("owner release should free the model")
	}
	if err := r.Acquire(path, "engine-b"); err != nil {
		t.Fatalf("acquire after release should succeed: %v", err)
	}
}

func TestMultiModelRegistry(t *testing.T) {
	r := NewMultiModelRegistry()
	r.Register(&ModelEntry{ModelName: "qwen3.5-4b", Aliases: []string{"q4"}, ToolCallParser: "hermes"})
	r.Register(&ModelEntry{ModelName: "llama-3", Aliases: []string{"l3"}})

	if r.Default() != "qwen3.5-4b" {
		t.Errorf("first registered should be default, got %q", r.Default())
	}
	// Empty name resolves to default.
	if e, ok := r.Resolve(""); !ok || e.ModelName != "qwen3.5-4b" {
		t.Errorf("empty resolve should hit default: %v %v", e, ok)
	}
	// Alias resolves to the same entry.
	if e, ok := r.Resolve("q4"); !ok || e.ToolCallParser != "hermes" {
		t.Errorf("alias resolve failed: %v %v", e, ok)
	}
	if e, ok := r.Resolve("l3"); !ok || e.ModelName != "llama-3" {
		t.Errorf("second model alias resolve failed: %v %v", e, ok)
	}
	if _, ok := r.Resolve("nope"); ok {
		t.Error("unknown model should not resolve")
	}
}
