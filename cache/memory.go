// SPDX-License-Identifier: Apache-2.0

package cache

// Memory-aware prefix caching sizes the resident set by how much key/value
// memory it occupies rather than by a fixed block count. Block cost depends on
// the model (layer count, head geometry, and dtype), so a count-based limit that
// is right for a small model either wastes memory or risks running out on a large
// one. The helpers here turn a model's shape and a memory budget into the per
// block byte cost and limit that the prefix cache evicts against.

const (
	// MinByteLimit is the floor a recommended budget is clamped to, so a tiny
	// percentage of a small machine still leaves room for at least a little
	// prefix reuse rather than a budget that holds nothing.
	MinByteLimit = int64(100 << 20) // 100 MiB

	// DefaultMemoryPercent is the share of a memory pool to spend on the prefix
	// cache when a caller does not specify one.
	DefaultMemoryPercent = 0.20
)

// BlockKVBytes returns the key/value memory one cached block occupies for a model
// with the given shape. A block holds blockSize token positions; each position
// stores a key and a value vector of kvHeads * headDim elements per layer, so the
// factor of two accounts for keys and values together. dtypeBytes is the size of
// one stored element, for example 2 for bf16 or fp16. A non-positive dimension
// yields zero, which disables memory tracking rather than charging nonsense.
func BlockKVBytes(layers, kvHeads, headDim, blockSize, dtypeBytes int) int64 {
	if layers <= 0 || kvHeads <= 0 || headDim <= 0 || blockSize <= 0 || dtypeBytes <= 0 {
		return 0
	}
	perPos := int64(layers) * 2 * int64(kvHeads) * int64(headDim) * int64(dtypeBytes)
	return perPos * int64(blockSize)
}

// RecommendByteLimit returns the share of poolBytes to spend on the cache, given
// a fraction in (0, 1]. A non-positive or out-of-range percent falls back to
// DefaultMemoryPercent. The result is clamped up to MinByteLimit so the cache is
// never sized down to nothing, but never above the pool itself.
func RecommendByteLimit(poolBytes int64, percent float64) int64 {
	if poolBytes <= 0 {
		return 0
	}
	if percent <= 0 || percent > 1 {
		percent = DefaultMemoryPercent
	}
	limit := int64(float64(poolBytes) * percent)
	limit = max(limit, MinByteLimit)
	limit = min(limit, poolBytes)
	return limit
}

// NewMemoryAwarePrefixCache returns a prefix cache that evicts to keep its
// resident key/value state at or below byteLimit, charging blockBytes per block.
// blockBytes is what BlockKVBytes returns for the model; byteLimit is what
// RecommendByteLimit returns for the memory pool. A non-positive byteLimit or
// blockBytes disables memory-based eviction, leaving the cache holding every
// block, so callers that want a bound must pass both.
func NewMemoryAwarePrefixCache(blockSize int, blockBytes, byteLimit int64) *PrefixCache {
	c := NewPrefixCache(blockSize, 0)
	if blockBytes > 0 && byteLimit > 0 {
		c.blockBytes = blockBytes
		c.byteLimit = byteLimit
	}
	return c
}

// Capacity reports the memory budget in bytes, or zero when the cache is not
// tracking memory.
func (c *PrefixCache) Capacity() int64 { return c.byteLimit }
