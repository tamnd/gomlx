// SPDX-License-Identifier: Apache-2.0

package doctor

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/gomlx/models"
)

// firstProfile returns a known alias and its repo id to build cache fixtures
// against, so the test does not hardcode a name that aliases.json might rename.
func firstProfile(t *testing.T) (alias, repoID string) {
	t.Helper()
	profiles := models.Profiles()
	if len(profiles) == 0 {
		t.Skip("no aliases registered")
	}
	return profiles[0].Alias, profiles[0].HFPath
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCacheRootsPriorityAndDedup(t *testing.T) {
	env := map[string]string{
		"HF_HUB_CACHE": "/cache/hub",
		"HF_HOME":      "/hf",
	}
	lookup := func(k string) (string, bool) { v, ok := env[k]; return v, ok }
	roots := CacheRoots(lookup, "/home/u")
	want := []string{
		"/cache/hub",
		filepath.Join("/hf", "hub"),
		filepath.Join("/home/u", ".cache", "huggingface", "hub"),
		filepath.Join("/home/u", ".lmstudio", "models"),
	}
	if len(roots) != len(want) {
		t.Fatalf("roots=%v want %v", roots, want)
	}
	for i := range want {
		if roots[i] != want[i] {
			t.Fatalf("root %d = %q want %q", i, roots[i], want[i])
		}
	}
}

func TestDiscoverFindsCompleteHFSnapshot(t *testing.T) {
	root := t.TempDir()
	_, repoID := firstProfile(t)
	snap := filepath.Join(root, repoToHFDirname(repoID), "snapshots", "abc123")
	write(t, filepath.Join(snap, "config.json"), "{}")
	write(t, filepath.Join(snap, "model.safetensors"), "weights")

	avail := AvailableAliases([]string{root})
	if len(avail) != 1 {
		t.Fatalf("available=%v want exactly the one fixture model", avail)
	}
}

func TestDiscoverReportsPartialDownload(t *testing.T) {
	root := t.TempDir()
	alias, repoID := firstProfile(t)
	snap := filepath.Join(root, repoToHFDirname(repoID), "snapshots", "abc123")
	write(t, filepath.Join(snap, "config.json"), "{}") // config but no weights

	for _, m := range DiscoverModels([]string{root}) {
		if m.Alias != alias {
			continue
		}
		if m.Available {
			t.Fatal("a config-only snapshot must not count as available")
		}
		if m.Reason == "" {
			t.Fatal("unavailable model should carry a reason")
		}
		return
	}
	t.Fatalf("alias %q not present in discovery output", alias)
}

func TestShardedSnapshotNeedsEveryShard(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "config.json"), "{}")
	write(t, filepath.Join(dir, "model.safetensors.index.json"),
		`{"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}`)
	write(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), "x")

	if isCompleteSnapshot(dir) {
		t.Fatal("snapshot missing a shard must be incomplete")
	}
	write(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), "y")
	if !isCompleteSnapshot(dir) {
		t.Fatal("snapshot with every shard should be complete")
	}
}

func TestUnreadableIndexIsIncomplete(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "config.json"), "{}")
	write(t, filepath.Join(dir, "model.safetensors.index.json"), "{ this is not json")
	write(t, filepath.Join(dir, "model.safetensors"), "x")
	if isCompleteSnapshot(dir) {
		t.Fatal("an unreadable index should be treated as incomplete, not fall through")
	}
}

func TestModelsCheckWarnsWhenEmpty(t *testing.T) {
	if r := ModelsCheck([]string{t.TempDir()}); r.Status != StatusWarn {
		t.Fatalf("empty cache should warn, got %s", r.Status)
	}
	root := t.TempDir()
	_, repoID := firstProfile(t)
	snap := filepath.Join(root, repoToHFDirname(repoID), "snapshots", "s")
	write(t, filepath.Join(snap, "config.json"), "{}")
	write(t, filepath.Join(snap, "model.safetensors"), "w")
	if r := ModelsCheck([]string{root}); r.Status != StatusPass {
		t.Fatalf("cache with a model should pass, got %s", r.Status)
	}
}
