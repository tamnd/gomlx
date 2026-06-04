// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestSnapshotRestoreRoundTrip(t *testing.T) {
	c := NewPrefixCache(4, 0)
	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	handles, _ := c.Insert(tokens)
	if len(handles) != 3 {
		t.Fatalf("expected 3 blocks, got %d", len(handles))
	}

	snap := c.Snapshot()
	if snap.BlockSize != 4 || len(snap.Blocks) != 3 {
		t.Fatalf("snapshot: blockSize=%d blocks=%d", snap.BlockSize, len(snap.Blocks))
	}

	restored := RestorePrefixCache(snap, 0)
	gotMatched, gotHandles := restored.Match(tokens)
	if gotMatched != 12 {
		t.Errorf("restored match length: got %d want 12", gotMatched)
	}
	if !reflect.DeepEqual(gotHandles, handles) {
		t.Errorf("restored handles: got %v want %v", gotHandles, handles)
	}

	// A new insert on the restored cache must mint a fresh handle, not collide
	// with a restored one.
	more, added := restored.Insert([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
	if !added[len(added)-1] {
		t.Error("the new fourth block should be freshly added")
	}
	for _, h := range handles {
		if more[len(more)-1] == h {
			t.Errorf("new handle %d collides with a restored handle", h)
		}
	}
}

func TestStoreSaveLoad(t *testing.T) {
	store := NewStore(t.TempDir())
	model := "qwen3-0.6b"
	fp := []byte(`{"hidden_size":1024}`)

	c := NewPrefixCache(4, 0)
	tokens := []int32{10, 20, 30, 40, 50, 60, 70, 80}
	handles, _ := c.Insert(tokens)
	snap := c.Snapshot()

	blobs := map[Handle][]byte{
		handles[0]: []byte("keys-and-values-of-block-0"),
		handles[1]: {0x00, 0x01, 0x02, 0xff, 0xfe},
	}

	if store.Has(model, fp) {
		t.Fatal("cache should not exist before save")
	}
	if err := store.Save(model, fp, snap, blobs); err != nil {
		t.Fatalf("save: %v", err)
	}
	if !store.Has(model, fp) {
		t.Fatal("cache should exist after save")
	}

	gotSnap, gotBlobs, err := store.Load(model, fp)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !reflect.DeepEqual(gotSnap, snap) {
		t.Errorf("snapshot round trip: got %+v want %+v", gotSnap, snap)
	}
	if !reflect.DeepEqual(gotBlobs, blobs) {
		t.Errorf("blobs round trip: got %v want %v", gotBlobs, blobs)
	}

	// The reloaded snapshot must rebuild a cache that still matches the prompt.
	restored := RestorePrefixCache(gotSnap, 0)
	if matched, _ := restored.Match(tokens); matched != 8 {
		t.Errorf("restored-from-disk match: got %d want 8", matched)
	}
}

func TestStoreFingerprintIsolatesModels(t *testing.T) {
	store := NewStore(t.TempDir())
	model := "qwen3-0.6b"

	// The same name with a different fingerprint is a different cache.
	if err := store.Save(model, []byte("weights-A"), Snapshot{BlockSize: 4}, nil); err != nil {
		t.Fatalf("save A: %v", err)
	}
	if store.Has(model, []byte("weights-B")) {
		t.Error("a different fingerprint must not see the first cache")
	}
	if !store.Has(model, []byte("weights-A")) {
		t.Error("the original fingerprint must still see its cache")
	}

	if store.Dir(model, []byte("weights-A")) == store.Dir(model, []byte("weights-B")) {
		t.Error("different fingerprints must map to different directories")
	}
}

func TestCacheKeyShape(t *testing.T) {
	key := CacheKey("org/Model:v1.5", []byte("fp"))
	// No path-unsafe characters survive in the sanitized name.
	name := strings.SplitN(key, "--", 2)[0]
	if strings.ContainsAny(name, "/:") {
		t.Errorf("sanitized name still has unsafe chars: %q", name)
	}
	// The suffix is eight hex digits and is deterministic.
	if CacheKey("org/Model:v1.5", []byte("fp")) != key {
		t.Error("CacheKey is not deterministic")
	}
	parts := strings.SplitN(key, "--", 2)
	if len(parts) != 2 || len(parts[1]) != 8 {
		t.Errorf("key %q does not end in an 8-char fingerprint", key)
	}
}

func TestSafeModelName(t *testing.T) {
	cases := map[string]string{
		"qwen3-0.6b":        "qwen3-0.6b",
		"org/model":         "org_model",
		"a:b c/d\\e":        "a_b_c_d_e",
		"":                  "model",
		"Meta-Llama_3.1-8B": "Meta-Llama_3.1-8B",
	}
	for in, want := range cases {
		if got := safeModelName(in); got != want {
			t.Errorf("safeModelName(%q): got %q want %q", in, got, want)
		}
	}
}

func TestStoreLoadMissing(t *testing.T) {
	store := NewStore(t.TempDir())
	if _, _, err := store.Load("absent", []byte("x")); err == nil {
		t.Error("loading a cache that was never saved must error")
	}
}

func TestDefaultCacheRootShape(t *testing.T) {
	root := DefaultCacheRoot()
	if filepath.Base(root) != "prefix_cache" {
		t.Errorf("default root should end in prefix_cache, got %q", root)
	}
}
