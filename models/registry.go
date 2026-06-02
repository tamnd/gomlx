// SPDX-License-Identifier: Apache-2.0

package models

import (
	"fmt"
	"sync"
)

// OwnershipRegistry prevents two engines from sharing a model's batch KV cache.
// The reference uses weakrefs; Go uses explicit acquire/release on engine
// start/stop. Ported from model_registry.py.
type OwnershipRegistry struct {
	mu     sync.Mutex
	owners map[string]string // hf_path -> owner id
}

// NewOwnershipRegistry returns an empty registry.
func NewOwnershipRegistry() *OwnershipRegistry {
	return &OwnershipRegistry{owners: map[string]string{}}
}

// Acquire claims ownership of a model path for an owner. It fails if another
// owner already holds it.
func (r *OwnershipRegistry) Acquire(hfPath, owner string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if cur, ok := r.owners[hfPath]; ok && cur != owner {
		return fmt.Errorf("model %q is already owned by %q", hfPath, cur)
	}
	r.owners[hfPath] = owner
	return nil
}

// Release drops ownership if held by owner; a mismatched owner is a no-op.
func (r *OwnershipRegistry) Release(hfPath, owner string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if cur, ok := r.owners[hfPath]; ok && cur == owner {
		delete(r.owners, hfPath)
	}
}

// Owner returns the current owner of a model path, or "" if none.
func (r *OwnershipRegistry) Owner(hfPath string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.owners[hfPath]
}

// ModelEntry describes one model served by a multi-model deployment.
type ModelEntry struct {
	Engine          any // *engine.BatchedEngine once wired (stage 2/4)
	ModelName       string
	Aliases         []string
	ToolCallParser  string
	ReasoningParser string
	IsMLLM          bool
	MaxTokens       int
}

// MultiModelRegistry maps a request's "model" field to the engine serving it,
// for multi-model deployments (Hermes / agent workflows). Ported from
// runtime/model_registry.py.
type MultiModelRegistry struct {
	mu      sync.RWMutex
	entries map[string]*ModelEntry // key: model name or alias
	def     string                 // default model key
}

// NewMultiModelRegistry returns an empty registry.
func NewMultiModelRegistry() *MultiModelRegistry {
	return &MultiModelRegistry{entries: map[string]*ModelEntry{}}
}

// Register adds an entry under its model name and every alias. The first entry
// registered becomes the default.
func (r *MultiModelRegistry) Register(e *ModelEntry) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.def == "" {
		r.def = e.ModelName
	}
	r.entries[e.ModelName] = e
	for _, a := range e.Aliases {
		r.entries[a] = e
	}
}

// Resolve returns the entry for a model name/alias. An empty name returns the
// default entry. The bool reports whether a match was found.
func (r *MultiModelRegistry) Resolve(name string) (*ModelEntry, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if name == "" {
		name = r.def
	}
	e, ok := r.entries[name]
	return e, ok
}

// Default returns the default model key.
func (r *MultiModelRegistry) Default() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.def
}
