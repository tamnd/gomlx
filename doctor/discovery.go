// SPDX-License-Identifier: Apache-2.0

package doctor

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/tamnd/gomlx/models"
)

// ModelAvailability says whether one alias has usable weights on disk.
type ModelAvailability struct {
	Alias     string
	RepoID    string
	Available bool
	Path      string // snapshot directory when Available
	Reason    string // why not, when unavailable
}

// CacheRoots returns the directories to search for downloaded weights, in
// priority order and de-duplicated. The Hugging Face hub cache wins, then HF_HOME,
// then the standard per-user cache, then an LM Studio store, since users often
// already have models in one of those. lookupEnv and home are injected so tests
// do not depend on the real environment.
func CacheRoots(lookupEnv func(string) (string, bool), home string) []string {
	var roots []string
	if v, ok := lookupEnv("HF_HUB_CACHE"); ok && v != "" {
		roots = append(roots, v)
	}
	if v, ok := lookupEnv("HF_HOME"); ok && v != "" {
		roots = append(roots, filepath.Join(v, "hub"))
	}
	if home != "" {
		roots = append(roots,
			filepath.Join(home, ".cache", "huggingface", "hub"),
			filepath.Join(home, ".lmstudio", "models"),
		)
	}
	seen := map[string]bool{}
	out := roots[:0]
	for _, r := range roots {
		if r != "" && !seen[r] {
			seen[r] = true
			out = append(out, r)
		}
	}
	return out
}

// DefaultCacheRoots resolves CacheRoots from the live environment.
func DefaultCacheRoots() []string {
	home, _ := os.UserHomeDir()
	return CacheRoots(os.LookupEnv, home)
}

// repoToHFDirname is how the HF hub names a repo's cache directory.
func repoToHFDirname(repoID string) string {
	return "models--" + strings.ReplaceAll(repoID, "/", "--")
}

// DiscoverModels reports availability for every known alias across the given
// cache roots, available ones first and then alphabetical. It never downloads
// anything; a partial download is reported as unavailable with a reason rather
// than counted as present.
func DiscoverModels(roots []string) []ModelAvailability {
	var out []ModelAvailability
	for _, p := range models.Profiles() {
		out = append(out, checkAlias(p.Alias, p.HFPath, roots))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Available != out[j].Available {
			return out[i].Available
		}
		return out[i].Alias < out[j].Alias
	})
	return out
}

// AvailableAliases is the subset of aliases with complete weights on disk.
func AvailableAliases(roots []string) []string {
	var out []string
	for _, m := range DiscoverModels(roots) {
		if m.Available {
			out = append(out, m.Alias)
		}
	}
	return out
}

func checkAlias(alias, repoID string, roots []string) ModelAvailability {
	hfDir := repoToHFDirname(repoID)
	sawPartial := false
	for _, root := range roots {
		// Hugging Face hub layout: {root}/models--org--repo/snapshots/<sha>/
		snapshots := filepath.Join(root, hfDir, "snapshots")
		if entries, err := os.ReadDir(snapshots); err == nil {
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				snap := filepath.Join(snapshots, e.Name())
				if isCompleteSnapshot(snap) {
					return ModelAvailability{Alias: alias, RepoID: repoID, Available: true, Path: snap}
				}
				if fileExists(filepath.Join(snap, "config.json")) {
					sawPartial = true
				}
			}
		}
		// LM Studio / raw layout: {root}/org/repo/
		raw := filepath.Join(root, filepath.Join(strings.Split(repoID, "/")...))
		if isCompleteSnapshot(raw) {
			return ModelAvailability{Alias: alias, RepoID: repoID, Available: true, Path: raw}
		}
		if fileExists(filepath.Join(raw, "config.json")) {
			sawPartial = true
		}
	}
	reason := "not found in any cache root"
	if sawPartial {
		reason = "found config.json but weights are missing, likely a partial download"
	}
	return ModelAvailability{Alias: alias, RepoID: repoID, Available: false, Reason: reason}
}

// isCompleteSnapshot reports whether dir holds a model that could actually load:
// a config.json plus weights. For a sharded model every shard named in the index
// must be present; otherwise any one weight file is enough. os.Stat follows
// symlinks, so a dangling link left by an interrupted download counts as missing.
func isCompleteSnapshot(dir string) bool {
	if !fileExists(filepath.Join(dir, "config.json")) {
		return false
	}
	index := filepath.Join(dir, "model.safetensors.index.json")
	if fileExists(index) {
		if shards, ok := shardsFromIndex(index); ok {
			if len(shards) == 0 {
				return hasAnyWeight(dir)
			}
			for shard := range shards {
				if !fileExists(filepath.Join(dir, shard)) {
					return false
				}
			}
			return true
		}
		// Index present but unreadable: do not trust the snapshot.
		return false
	}
	return hasAnyWeight(dir)
}

// shardsFromIndex returns the set of shard filenames the index references. The
// bool is false when the index cannot be read or parsed.
func shardsFromIndex(path string) (map[string]struct{}, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, false
	}
	shards := map[string]struct{}{}
	for _, shard := range index.WeightMap {
		shards[shard] = struct{}{}
	}
	return shards, true
}

func hasAnyWeight(dir string) bool {
	for _, ext := range []string{"*.safetensors", "*.npz", "*.gguf"} {
		matches, err := filepath.Glob(filepath.Join(dir, ext))
		if err != nil {
			continue
		}
		if slices.ContainsFunc(matches, fileExists) {
			return true
		}
	}
	return false
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
