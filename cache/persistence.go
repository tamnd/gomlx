// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// Persisting a prefix cache lets a server skip recomputing a long shared prompt
// after a restart: the system prompt or few-shot preamble that every request
// shares is computed once, written to disk, and reloaded on the next run. What
// persists is the prefix index (which block hashes are cached, in which order)
// alongside the key/value bytes of each block. On load the index restores
// directly; the compute layer reloads each block's bytes onto the device and
// re-registers the handle.
//
// A cache is only valid for the exact model it was built from, so each cache
// lives in a directory keyed by the model name and a fingerprint of the model
// (its config or weights). A different model, or the same model with changed
// weights, lands in a different directory and never reuses stale state.

// SnapBlock is one resident block in a persisted snapshot: the chained hash that
// identifies its prefix and the handle the compute layer uses to find its
// key/value bytes.
type SnapBlock struct {
	Hash   uint64
	Handle Handle
}

// Snapshot is a serializable view of a prefix cache's resident blocks in
// most-recently-used-first order, enough to rebuild an equivalent cache.
type Snapshot struct {
	BlockSize int
	Blocks    []SnapBlock
}

// Snapshot exports the resident blocks in LRU order, front (most recently used)
// first. Pin state is not exported: a reloaded cache starts with no in-flight
// readers, which is correct after a restart.
func (c *PrefixCache) Snapshot() Snapshot {
	snap := Snapshot{BlockSize: c.blockSize}
	for e := c.lru.Front(); e != nil; e = e.Next() {
		b := e.Value.(*block)
		snap.Blocks = append(snap.Blocks, SnapBlock{Hash: b.hash, Handle: b.handle})
	}
	return snap
}

// RestorePrefixCache rebuilds a cache from a snapshot, preserving block order and
// handles so reloaded key/value state keeps its identity. capacity sets the
// resident block limit for the restored cache, as in NewPrefixCache. New inserts
// continue from the highest restored handle so they never collide.
func RestorePrefixCache(snap Snapshot, capacity int) *PrefixCache {
	c := NewPrefixCache(snap.BlockSize, capacity)
	for _, sb := range snap.Blocks {
		b := &block{hash: sb.Hash, handle: sb.Handle}
		b.elem = c.lru.PushBack(b) // forward order keeps Blocks[0] at the front
		c.blocks[sb.Hash] = b
		if sb.Handle > c.nextID {
			c.nextID = sb.Handle
		}
	}
	return c
}

// Store reads and writes prefix-cache snapshots under a root directory, one
// subdirectory per model. It is safe to point several servers at the same root;
// each model's state is isolated by its keyed directory.
type Store struct {
	root string
}

// NewStore returns a store rooted at dir. DefaultCacheRoot supplies the
// conventional location when a caller has no preference.
func NewStore(dir string) *Store {
	return &Store{root: dir}
}

// DefaultCacheRoot is the conventional cache location,
// ~/.cache/gomlx/prefix_cache. It falls back to a relative path if the home
// directory cannot be determined.
func DefaultCacheRoot() string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return filepath.Join(".cache", "gomlx", "prefix_cache")
	}
	return filepath.Join(home, ".cache", "gomlx", "prefix_cache")
}

// CacheKey is the per-model directory name: a sanitized model name, two dashes,
// and the first eight hex digits of the SHA-256 of the fingerprint. The
// fingerprint is whatever uniquely identifies the model's weights, typically the
// raw config or a weights digest; any change to it routes to a fresh directory.
func CacheKey(model string, fingerprint []byte) string {
	sum := sha256.Sum256(fingerprint)
	return fmt.Sprintf("%s--%s", safeModelName(model), hex.EncodeToString(sum[:])[:8])
}

// Dir returns the absolute directory a model's cache lives in under the store.
func (s *Store) Dir(model string, fingerprint []byte) string {
	return filepath.Join(s.root, CacheKey(model, fingerprint))
}

// Has reports whether a saved cache exists for the model and fingerprint.
func (s *Store) Has(model string, fingerprint []byte) bool {
	_, err := os.Stat(filepath.Join(s.Dir(model, fingerprint), manifestName))
	return err == nil
}

const manifestName = "manifest.json"

// diskManifest is the on-disk form of a snapshot. Hashes and handles are written
// as strings because they are 64-bit and JSON numbers cannot carry the full
// range without loss.
type diskManifest struct {
	BlockSize int             `json:"block_size"`
	Blocks    []diskSnapBlock `json:"blocks"`
}

type diskSnapBlock struct {
	Hash   string `json:"hash"`
	Handle string `json:"handle"`
}

// Save writes a snapshot and its per-handle key/value blobs to the model's
// directory, replacing any previous contents of that directory. blobs maps each
// block handle in the snapshot to the opaque bytes the compute layer produced
// for it; the store treats them as opaque so the byte format stays the compute
// layer's concern.
func (s *Store) Save(model string, fingerprint []byte, snap Snapshot, blobs map[Handle][]byte) error {
	dir := s.Dir(model, fingerprint)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("cache: create dir: %w", err)
	}

	man := diskManifest{BlockSize: snap.BlockSize}
	for _, b := range snap.Blocks {
		man.Blocks = append(man.Blocks, diskSnapBlock{
			Hash:   strconv.FormatUint(b.Hash, 16),
			Handle: strconv.FormatUint(uint64(b.Handle), 16),
		})
	}
	data, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return fmt.Errorf("cache: encode manifest: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, manifestName), data, 0o644); err != nil {
		return fmt.Errorf("cache: write manifest: %w", err)
	}

	for h, blob := range blobs {
		if err := os.WriteFile(filepath.Join(dir, blobName(h)), blob, 0o644); err != nil {
			return fmt.Errorf("cache: write blob %d: %w", h, err)
		}
	}
	return nil
}

// Load reads back a snapshot and its blobs for the model and fingerprint. It
// returns the snapshot in saved order and the blobs keyed by handle. A missing
// cache is an error; callers that tolerate a cold start should check Has first.
func (s *Store) Load(model string, fingerprint []byte) (Snapshot, map[Handle][]byte, error) {
	dir := s.Dir(model, fingerprint)
	data, err := os.ReadFile(filepath.Join(dir, manifestName))
	if err != nil {
		return Snapshot{}, nil, fmt.Errorf("cache: read manifest: %w", err)
	}
	var man diskManifest
	if err := json.Unmarshal(data, &man); err != nil {
		return Snapshot{}, nil, fmt.Errorf("cache: decode manifest: %w", err)
	}

	snap := Snapshot{BlockSize: man.BlockSize}
	blobs := make(map[Handle][]byte, len(man.Blocks))
	for _, db := range man.Blocks {
		hash, err := strconv.ParseUint(db.Hash, 16, 64)
		if err != nil {
			return Snapshot{}, nil, fmt.Errorf("cache: bad block hash %q: %w", db.Hash, err)
		}
		handleU, err := strconv.ParseUint(db.Handle, 16, 64)
		if err != nil {
			return Snapshot{}, nil, fmt.Errorf("cache: bad block handle %q: %w", db.Handle, err)
		}
		h := Handle(handleU)
		snap.Blocks = append(snap.Blocks, SnapBlock{Hash: hash, Handle: h})

		blob, err := os.ReadFile(filepath.Join(dir, blobName(h)))
		if err != nil {
			return Snapshot{}, nil, fmt.Errorf("cache: read blob %d: %w", h, err)
		}
		blobs[h] = blob
	}
	return snap, blobs, nil
}

// blobName is the file name a block's key/value bytes are stored under.
func blobName(h Handle) string {
	return "blob-" + strconv.FormatUint(uint64(h), 16) + ".kv"
}

// safeModelName turns a model name into a single safe directory segment by
// replacing any character outside a conservative set with an underscore.
// Uniqueness across names that collide under this map is restored by the
// fingerprint suffix CacheKey appends.
func safeModelName(name string) string {
	if name == "" {
		return "model"
	}
	out := []byte(name)
	for i, c := range out {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '.' || c == '_' || c == '-':
		default:
			out[i] = '_'
		}
	}
	return string(out)
}
