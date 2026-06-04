// SPDX-License-Identifier: Apache-2.0

package vision

import (
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"sync"
)

// Running an image through a vision tower is expensive, and the same image
// recurs constantly: a multi-turn chat resends the whole history, including any
// attached image, on every request. Encoding it once and reusing the result
// across turns removes that repeated work. EmbedCache keys a tower's output by a
// hash of the exact image bytes and the preprocessing that produced them, so a
// byte-identical image under the same config hits the cache while a re-encoded or
// differently preprocessed image misses it.

// Embedding is a vision tower's output for one image: a flat tensor with its
// shape. The layout is the tower's concern; the cache treats Data as opaque
// values it stores and returns unchanged.
type Embedding struct {
	Shape []int
	Data  []float32
}

// EmbedKey identifies a cached embedding. It is the SHA-256 of the preprocessing
// fingerprint followed by the raw image bytes, so two requests that send the same
// image under the same config share a key and any difference in either routes to
// a different key.
type EmbedKey [32]byte

// KeyFor computes the cache key for an image under a preprocessing config. The
// config is folded in because the same bytes preprocessed differently yield a
// different embedding and must not collide.
func KeyFor(imageData []byte, cfg PreprocessConfig) EmbedKey {
	h := sha256.New()
	h.Write(configFingerprint(cfg))
	h.Write(imageData)
	var k EmbedKey
	copy(k[:], h.Sum(nil))
	return k
}

// configFingerprint serializes the fields of a config that change the output into
// a stable byte string. Width and height change the token count; the rescale,
// mean, and std change the values; all of them must distinguish keys.
func configFingerprint(cfg PreprocessConfig) []byte {
	buf := make([]byte, 0, 4+4+4*9)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(cfg.Width))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(cfg.Height))
	buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(cfg.RescaleFactor))
	for c := range 3 {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(cfg.Mean[c]))
	}
	for c := range 3 {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(cfg.Std[c]))
	}
	return buf
}

// EmbedCache is a fixed-capacity, least-recently-used cache of vision embeddings.
// It is safe for concurrent use: several request handlers may share one cache and
// look up images in parallel. Capacity bounds the number of resident embeddings;
// the least recently used entry is evicted when a new one would exceed it.
type EmbedCache struct {
	mu       sync.Mutex
	capacity int
	lru      *list.List // front = most recently used
	entries  map[EmbedKey]*list.Element
}

type embedEntry struct {
	key   EmbedKey
	embed *Embedding
}

// NewEmbedCache returns a cache holding at most capacity embeddings. A capacity of
// zero or less disables caching: Put is a no-op and Get always misses, which lets
// a caller turn the cache off without branching on a nil cache.
func NewEmbedCache(capacity int) *EmbedCache {
	return &EmbedCache{
		capacity: capacity,
		lru:      list.New(),
		entries:  make(map[EmbedKey]*list.Element),
	}
}

// Get returns the cached embedding for a key and reports whether it was present.
// A hit moves the entry to the most-recently-used position so it survives longer
// under eviction.
func (c *EmbedCache) Get(key EmbedKey) (*Embedding, bool) {
	if c.capacity <= 0 {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.entries[key]
	if !ok {
		return nil, false
	}
	c.lru.MoveToFront(el)
	return el.Value.(*embedEntry).embed, true
}

// Put stores an embedding under a key, evicting the least recently used entry if
// the cache is at capacity. Storing a key that is already present refreshes its
// value and marks it most recently used rather than adding a duplicate.
func (c *EmbedCache) Put(key EmbedKey, embed *Embedding) {
	if c.capacity <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if el, ok := c.entries[key]; ok {
		el.Value.(*embedEntry).embed = embed
		c.lru.MoveToFront(el)
		return
	}

	el := c.lru.PushFront(&embedEntry{key: key, embed: embed})
	c.entries[key] = el

	for c.lru.Len() > c.capacity {
		oldest := c.lru.Back()
		if oldest == nil {
			break
		}
		c.lru.Remove(oldest)
		delete(c.entries, oldest.Value.(*embedEntry).key)
	}
}

// Len reports the number of resident embeddings.
func (c *EmbedCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lru.Len()
}
