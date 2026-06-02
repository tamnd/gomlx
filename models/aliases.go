// SPDX-License-Identifier: Apache-2.0

// Package models holds the model alias registry, auto-config detection, and the
// runtime model registry.
package models

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

//go:embed aliases.json
var aliasesJSON []byte

// AliasProfile is a single entry in aliases.json. Field set matches the union
// of all keys present in the reference file.
type AliasProfile struct {
	HFPath              string             `json:"hf_path"`
	ToolCallParser      string             `json:"tool_call_parser,omitempty"`
	ReasoningParser     string             `json:"reasoning_parser,omitempty"`
	IsHybrid            bool               `json:"is_hybrid,omitempty"`
	IsMoE               bool               `json:"is_moe,omitempty"`
	SupportsSpecDecode  bool               `json:"supports_spec_decode,omitempty"`
	SuffixDecodingTier  string             `json:"suffix_decoding_tier,omitempty"`
	SuffixBenchSpeedup  map[string]float64 `json:"suffix_bench_speedup,omitempty"`
	SupportsDFlash      bool               `json:"supports_dflash,omitempty"`
	DFlashDraftModel    string             `json:"dflash_draft_model,omitempty"`
	RecommendedSampling map[string]float64 `json:"recommended_sampling,omitempty"`

	// Alias is the lookup key; populated at load time, not from JSON.
	Alias string `json:"-"`
}

var (
	aliasIndex map[string]AliasProfile // alias -> profile
	pathIndex  map[string]string       // hf_path -> alias
)

func init() {
	raw := map[string]AliasProfile{}
	if err := json.Unmarshal(aliasesJSON, &raw); err != nil {
		panic(fmt.Sprintf("models: malformed embedded aliases.json: %v", err))
	}
	aliasIndex = make(map[string]AliasProfile, len(raw))
	pathIndex = make(map[string]string, len(raw))
	for alias, p := range raw {
		p.Alias = alias
		aliasIndex[alias] = p
		// First alias wins for a given hf_path (deterministic via sort below).
		if _, ok := pathIndex[p.HFPath]; !ok {
			pathIndex[p.HFPath] = alias
		}
	}
	// Deterministic reverse index: prefer the lexically-smallest alias per path.
	pathIndex = map[string]string{}
	aliases := make([]string, 0, len(aliasIndex))
	for a := range aliasIndex {
		aliases = append(aliases, a)
	}
	sort.Strings(aliases)
	for _, a := range aliases {
		path := aliasIndex[a].HFPath
		if _, ok := pathIndex[path]; !ok {
			pathIndex[path] = a
		}
	}
}

// ResolveModel maps an alias (or a raw hf_path) to its hf_path. If name is not a
// known alias it is returned unchanged (treated as a direct path).
func ResolveModel(name string) string {
	if p, ok := aliasIndex[name]; ok {
		return p.HFPath
	}
	return name
}

// ResolveProfile returns the AliasProfile for an alias or hf_path, plus whether
// a profile was found.
func ResolveProfile(name string) (AliasProfile, bool) {
	if p, ok := aliasIndex[name]; ok {
		return p, true
	}
	if alias, ok := pathIndex[name]; ok {
		return aliasIndex[alias], true
	}
	return AliasProfile{}, false
}

// AliasForPath returns the canonical alias for an hf_path, or "" if unknown.
func AliasForPath(hfPath string) string { return pathIndex[hfPath] }

// Aliases returns all known aliases, sorted.
func Aliases() []string {
	out := make([]string, 0, len(aliasIndex))
	for a := range aliasIndex {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}

// Profiles returns all profiles, sorted by alias.
func Profiles() []AliasProfile {
	out := make([]AliasProfile, 0, len(aliasIndex))
	for _, a := range Aliases() {
		out = append(out, aliasIndex[a])
	}
	return out
}

// IsAlias reports whether name is a known alias (not a raw path).
func IsAlias(name string) bool {
	_, ok := aliasIndex[name]
	return ok
}

// normalizeName lowercases and strips surrounding whitespace for matching.
func normalizeName(s string) string { return strings.ToLower(strings.TrimSpace(s)) }
