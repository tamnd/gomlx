// SPDX-License-Identifier: Apache-2.0

package compute

// This file holds the pure-Go bookkeeping for the attention KV caches: where
// the next keys and values land, how many positions are live, and which slots a
// rotating (sliding-window) cache keeps or evicts. The tensor storage and the
// concatenation itself live on-device behind the "mlx" build tag; everything
// here is the index math that decides the offsets, so it runs and is tested
// without a GPU. Keeping the arithmetic separate is what lets the scheduler
// reason about cache occupancy before any Metal kernel runs.

// KVCache tracks a growing key/value cache for one attention layer. It grows in
// steps of Step positions so the backing tensor is reallocated in chunks rather
// than on every token, matching the reference cache.
type KVCache struct {
	Offset int // number of valid positions currently stored
	Step   int // growth granularity; <=0 means grow exactly by the update
}

// DefaultStep is the allocation granularity used when Step is unset.
const DefaultStep = 256

// step returns the effective growth granularity.
func (c *KVCache) step() int {
	if c.Step <= 0 {
		return DefaultStep
	}
	return c.Step
}

// Reserve records that n new positions are being appended and returns the slot
// range [start, end) they occupy and the capacity the backing tensor must have
// to hold them. capacity is rounded up to a multiple of the step.
func (c *KVCache) Reserve(n int) (start, end, capacity int) {
	start = c.Offset
	c.Offset += n
	end = c.Offset
	capacity = roundUp(c.Offset, c.step())
	return start, end, capacity
}

// Trim drops the last n positions (for example after rejected speculative
// tokens), clamping at zero. It returns the number actually removed.
func (c *KVCache) Trim(n int) int {
	if n > c.Offset {
		n = c.Offset
	}
	if n < 0 {
		n = 0
	}
	c.Offset -= n
	return n
}

// Reset empties the cache.
func (c *KVCache) Reset() { c.Offset = 0 }

// RotatingKVCache is a fixed-window cache for sliding-window attention. It keeps
// at most MaxSize positions; Keep of the earliest positions (typically attention
// sinks) are pinned and never evicted. Once full, new positions overwrite the
// oldest non-pinned slot in a ring.
type RotatingKVCache struct {
	MaxSize int // window capacity, including the pinned prefix
	Keep    int // number of leading positions that are never evicted
	Offset  int // total positions seen (logical clock, only ever increases)
	size    int // positions currently resident (<= MaxSize)
}

// Len returns how many positions are currently resident.
func (c *RotatingKVCache) Len() int { return c.size }

// Update records one new position and returns the ring slot it is written to.
// Before the window fills, that is simply the next free slot. After it fills,
// writes cycle through the slots after the pinned prefix, evicting the oldest.
func (c *RotatingKVCache) Update() (slot int) {
	if c.MaxSize <= 0 {
		// Unbounded: behave like a plain append.
		slot = c.Offset
		c.Offset++
		c.size = c.Offset
		return slot
	}
	if c.size < c.MaxSize {
		slot = c.size
		c.size++
		c.Offset++
		return slot
	}
	// Window full: overwrite the oldest non-pinned slot in a ring of the
	// positions after the pinned prefix.
	span := c.MaxSize - c.Keep
	if span <= 0 {
		// Everything is pinned; nothing can be overwritten.
		c.Offset++
		return c.MaxSize - 1
	}
	slot = c.Keep + ((c.Offset - c.Keep) % span)
	c.Offset++
	return slot
}

// Reset empties the rotating cache.
func (c *RotatingKVCache) Reset() {
	c.Offset = 0
	c.size = 0
}

// roundUp returns the smallest multiple of step that is >= n. A non-positive
// step returns n unchanged.
func roundUp(n, step int) int {
	if step <= 0 {
		return n
	}
	if n%step == 0 {
		return n
	}
	return (n/step + 1) * step
}
