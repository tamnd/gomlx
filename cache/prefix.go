// SPDX-License-Identifier: Apache-2.0

// Package cache holds the serving-side caches that let the engine avoid
// recomputing work it has already done. The first of these is the prefix cache:
// when many requests share a leading run of tokens, typically a common system
// prompt or a few-shot preamble, the key/value state for that run only has to be
// computed once and can be reused by every later request that starts the same
// way. This is one of the largest throughput wins under real load, where the
// shared prefix is often far longer than the part of the prompt that differs.
package cache

import "container/list"

// Handle is an opaque identifier the prefix cache assigns to a reusable block of
// key/value state. The compute layer maps a handle to the on-device tensor that
// holds that block's keys and values; the cache itself only tracks which blocks
// exist, how they chain together, and which may be evicted.
type Handle uint64

// block is one cached run of BlockSize tokens. Its hash chains in the hash of
// every block before it, so a block is only ever reused when the whole prefix
// leading up to it matches. refs counts in-flight requests reading the block; a
// block with live readers is pinned and never evicted.
type block struct {
	hash   uint64
	handle Handle
	refs   int
	elem   *list.Element // position in the LRU list, most-recent at the front
}

// PrefixCache maps shared prompt prefixes to reusable key/value blocks. Tokens
// are grouped into fixed-size blocks; a block is keyed by a hash that chains its
// own tokens together with the hash of the block before it. Matching a new
// prompt walks its blocks in order and stops at the first that is not cached, so
// the result is always a genuine prefix. Eviction is least-recently-used over
// the blocks that no in-flight request is holding.
//
// PrefixCache is not safe for concurrent use; the engine owns one and drives it
// from the single scheduler goroutine, the same goroutine that owns the device.
type PrefixCache struct {
	blockSize int
	capacity  int // maximum resident blocks; <= 0 disables eviction
	nextID    Handle
	blocks    map[uint64]*block
	lru       *list.List // *block, front = most recently used

	hits   uint64
	misses uint64
}

// NewPrefixCache returns a cache that groups tokens into blocks of blockSize and
// keeps at most capacity blocks resident. A capacity of zero or less keeps every
// block (no eviction). blockSize must be positive.
func NewPrefixCache(blockSize, capacity int) *PrefixCache {
	if blockSize <= 0 {
		blockSize = 16
	}
	return &PrefixCache{
		blockSize: blockSize,
		capacity:  capacity,
		blocks:    make(map[uint64]*block),
		lru:       list.New(),
	}
}

// BlockSize reports the token granularity of the cache.
func (c *PrefixCache) BlockSize() int { return c.blockSize }

// Len reports how many blocks are currently resident.
func (c *PrefixCache) Len() int { return len(c.blocks) }

// Match returns the longest cached prefix of tokens, expressed as the number of
// matched tokens (always a multiple of the block size) and the handle of each
// matched block in order. Matched blocks are moved to the front of the LRU list
// so a reused prefix stays warm. The trailing partial block, if any, is never
// matched because only whole blocks carry cached state.
func (c *PrefixCache) Match(tokens []int32) (matched int, handles []Handle) {
	var parent uint64
	for i := 0; i+c.blockSize <= len(tokens); i += c.blockSize {
		h := blockHash(parent, tokens[i:i+c.blockSize])
		b, ok := c.blocks[h]
		if !ok {
			break
		}
		c.touch(b)
		handles = append(handles, b.handle)
		matched += c.blockSize
		parent = h
	}
	if matched > 0 {
		c.hits++
	} else {
		c.misses++
	}
	return matched, handles
}

// Insert registers every whole block of tokens that is not already cached and
// returns a handle for each block in order, including blocks that already
// existed. Inserting may evict the least-recently-used unpinned blocks to stay
// within capacity.
//
// The returned added slice marks, per block, whether the handle was freshly
// created (true) or reused from an existing entry (false), so the caller knows
// which blocks' on-device key/value state it must still compute and fill.
func (c *PrefixCache) Insert(tokens []int32) (handles []Handle, added []bool) {
	var parent uint64
	for i := 0; i+c.blockSize <= len(tokens); i += c.blockSize {
		h := blockHash(parent, tokens[i:i+c.blockSize])
		b, ok := c.blocks[h]
		if !ok {
			c.nextID++
			b = &block{hash: h, handle: c.nextID}
			b.elem = c.lru.PushFront(b)
			c.blocks[h] = b
			c.evict()
			added = append(added, true)
		} else {
			c.touch(b)
			added = append(added, false)
		}
		handles = append(handles, b.handle)
		parent = h
	}
	return handles, added
}

// Pin marks the blocks behind handles as in use so they survive eviction while a
// request decodes against them. Each Pin must be balanced by an Unpin.
func (c *PrefixCache) Pin(handles []Handle) {
	for _, h := range handles {
		if b := c.byHandle(h); b != nil {
			b.refs++
		}
	}
}

// Unpin releases a previous Pin, allowing the blocks to be evicted again once no
// other request holds them.
func (c *PrefixCache) Unpin(handles []Handle) {
	for _, h := range handles {
		if b := c.byHandle(h); b != nil && b.refs > 0 {
			b.refs--
		}
	}
}

// Stats reports cumulative prefix hits and misses, one count per Match call.
func (c *PrefixCache) Stats() (hits, misses uint64) { return c.hits, c.misses }

// touch moves a block to the front of the LRU list.
func (c *PrefixCache) touch(b *block) { c.lru.MoveToFront(b.elem) }

// evict drops least-recently-used unpinned blocks until the cache is within
// capacity. A capacity of zero or less disables eviction entirely.
func (c *PrefixCache) evict() {
	if c.capacity <= 0 {
		return
	}
	for len(c.blocks) > c.capacity {
		e := c.lru.Back()
		if e == nil {
			return
		}
		// Walk back past pinned blocks to find an evictable one. If every
		// resident block is pinned there is nothing to drop, so stop.
		victim := e
		for victim != nil && victim.Value.(*block).refs > 0 {
			victim = victim.Prev()
		}
		if victim == nil {
			return
		}
		b := victim.Value.(*block)
		c.lru.Remove(victim)
		delete(c.blocks, b.hash)
	}
}

// byHandle finds a resident block by its handle. The cache is small relative to
// a request's block count, so a linear scan is fine and avoids a second index.
func (c *PrefixCache) byHandle(h Handle) *block {
	for _, b := range c.blocks {
		if b.handle == h {
			return b
		}
	}
	return nil
}

// blockHash chains the hash of the preceding blocks (parent) with the tokens of
// this block using FNV-1a, so two blocks with the same tokens but a different
// prefix hash differently and never alias. Folding the parent in first is what
// makes the result depend on the whole prefix, not just this block's tokens.
func blockHash(parent uint64, tokens []int32) uint64 {
	const (
		offset64 = uint64(1469598103934665603)
		prime64  = uint64(1099511628211)
	)
	h := offset64
	h ^= parent
	h *= prime64
	for _, t := range tokens {
		u := uint32(t)
		for s := 0; s < 32; s += 8 {
			h ^= uint64((u >> s) & 0xff)
			h *= prime64
		}
	}
	return h
}
