// SPDX-License-Identifier: Apache-2.0

package vision

import (
	"sync"
	"testing"
)

func emb(v float32) *Embedding {
	return &Embedding{Shape: []int{1, 1}, Data: []float32{v}}
}

func TestKeyForIsContentAddressed(t *testing.T) {
	cfg := DefaultPreprocessConfig()
	a := KeyFor([]byte("image-bytes-A"), cfg)
	again := KeyFor([]byte("image-bytes-A"), cfg)
	b := KeyFor([]byte("image-bytes-B"), cfg)

	if a != again {
		t.Error("the same bytes and config must produce the same key")
	}
	if a == b {
		t.Error("different bytes must produce different keys")
	}
}

func TestKeyForFoldsConfig(t *testing.T) {
	data := []byte("same-bytes")
	base := DefaultPreprocessConfig()

	bigger := base
	bigger.Width = base.Width * 2
	if KeyFor(data, base) == KeyFor(data, bigger) {
		t.Error("a different target size must change the key")
	}

	shifted := base
	shifted.Mean[0] += 0.1
	if KeyFor(data, base) == KeyFor(data, shifted) {
		t.Error("a different normalization mean must change the key")
	}
}

func TestEmbedCacheGetPut(t *testing.T) {
	c := NewEmbedCache(4)
	key := KeyFor([]byte("img"), DefaultPreprocessConfig())

	if _, ok := c.Get(key); ok {
		t.Error("a fresh cache must miss")
	}
	c.Put(key, emb(1.5))
	got, ok := c.Get(key)
	if !ok {
		t.Fatal("a stored key must hit")
	}
	if got.Data[0] != 1.5 {
		t.Errorf("value: got %v want 1.5", got.Data[0])
	}
}

func TestEmbedCacheEvictsLRU(t *testing.T) {
	c := NewEmbedCache(2)
	k1 := KeyFor([]byte("1"), DefaultPreprocessConfig())
	k2 := KeyFor([]byte("2"), DefaultPreprocessConfig())
	k3 := KeyFor([]byte("3"), DefaultPreprocessConfig())

	c.Put(k1, emb(1))
	c.Put(k2, emb(2))
	// Touch k1 so k2 becomes the least recently used.
	if _, ok := c.Get(k1); !ok {
		t.Fatal("k1 should still be resident")
	}
	c.Put(k3, emb(3)) // evicts k2

	if _, ok := c.Get(k2); ok {
		t.Error("k2 should have been evicted as least recently used")
	}
	if _, ok := c.Get(k1); !ok {
		t.Error("k1 should survive: it was used more recently")
	}
	if _, ok := c.Get(k3); !ok {
		t.Error("k3 should be resident: it was just inserted")
	}
	if c.Len() != 2 {
		t.Errorf("len: got %d want 2", c.Len())
	}
}

func TestEmbedCacheUpdateNoDuplicate(t *testing.T) {
	c := NewEmbedCache(2)
	key := KeyFor([]byte("x"), DefaultPreprocessConfig())
	c.Put(key, emb(1))
	c.Put(key, emb(2)) // same key, new value

	if c.Len() != 1 {
		t.Errorf("re-putting a key must not add a duplicate, len=%d", c.Len())
	}
	got, _ := c.Get(key)
	if got.Data[0] != 2 {
		t.Errorf("re-put must refresh the value, got %v", got.Data[0])
	}
}

func TestEmbedCacheDisabled(t *testing.T) {
	c := NewEmbedCache(0)
	key := KeyFor([]byte("x"), DefaultPreprocessConfig())
	c.Put(key, emb(1))
	if _, ok := c.Get(key); ok {
		t.Error("a zero-capacity cache must never hit")
	}
	if c.Len() != 0 {
		t.Errorf("a disabled cache must stay empty, len=%d", c.Len())
	}
}

func TestEmbedCacheConcurrent(t *testing.T) {
	c := NewEmbedCache(64)
	cfg := DefaultPreprocessConfig()
	var wg sync.WaitGroup
	for i := range 32 {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			key := KeyFor([]byte{byte(n)}, cfg)
			c.Put(key, emb(float32(n)))
			if got, ok := c.Get(key); ok && got.Data[0] != float32(n) {
				t.Errorf("concurrent value mismatch for %d: got %v", n, got.Data[0])
			}
		}(i)
	}
	wg.Wait()
	if c.Len() != 32 {
		t.Errorf("all 32 distinct keys should fit under capacity 64, len=%d", c.Len())
	}
}
