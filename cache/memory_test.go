// SPDX-License-Identifier: Apache-2.0

package cache

import "testing"

func TestBlockKVBytes(t *testing.T) {
	// 2 layers, 4 kv heads, 8 head dim, 16 tokens, 2 bytes per element.
	// per position = 2 * 2 * 4 * 8 * 2 = 256; per block = 256 * 16 = 4096.
	got := BlockKVBytes(2, 4, 8, 16, 2)
	if got != 4096 {
		t.Fatalf("BlockKVBytes = %d want 4096", got)
	}
	if BlockKVBytes(0, 4, 8, 16, 2) != 0 {
		t.Fatal("a non-positive dimension should disable tracking by returning 0")
	}
}

func TestRecommendByteLimit(t *testing.T) {
	pool := int64(8 << 30) // 8 GiB
	if got := RecommendByteLimit(pool, 0.25); got != pool/4 {
		t.Fatalf("25%% of 8GiB = %d want %d", got, pool/4)
	}
	// Out-of-range percent falls back to the default share.
	if got := RecommendByteLimit(pool, 5); got != int64(float64(pool)*DefaultMemoryPercent) {
		t.Fatalf("bad percent should use default, got %d", got)
	}
	// A tiny share of a small pool is clamped up to the floor.
	if got := RecommendByteLimit(200<<20, 0.01); got != MinByteLimit {
		t.Fatalf("tiny share should clamp to floor %d, got %d", MinByteLimit, got)
	}
	// The limit never exceeds the pool itself.
	if got := RecommendByteLimit(50<<20, 0.5); got != 50<<20 {
		t.Fatalf("limit above a small pool should clamp to the pool, got %d", got)
	}
	if RecommendByteLimit(0, 0.2) != 0 {
		t.Fatal("no pool means no budget")
	}
}

// TestMemoryAwareEvictsToBudget inserts more blocks than the budget allows and
// checks the cache evicts down to fit, keeping the most recent blocks.
func TestMemoryAwareEvictsToBudget(t *testing.T) {
	const blockSize = 4
	blockBytes := int64(1000)
	// Budget holds at most three blocks (3000 bytes < 4 * 1000).
	c := NewMemoryAwarePrefixCache(blockSize, blockBytes, 3500)

	// Five distinct blocks worth of tokens.
	c.Insert(seq(blockSize * 5))

	if c.Len() != 3 {
		t.Fatalf("resident blocks = %d want 3 (budget bound)", c.Len())
	}
	if c.UsageBytes() != 3000 {
		t.Fatalf("usage = %d want 3000", c.UsageBytes())
	}
	if c.UsageBytes() > c.Capacity() {
		t.Fatalf("usage %d over budget %d", c.UsageBytes(), c.Capacity())
	}
}

// TestMemoryAwareKeepsPinnedOverBudget checks a pinned block is not evicted even
// when that pushes usage past the budget, since a live reader still needs it.
func TestMemoryAwareKeepsPinnedOverBudget(t *testing.T) {
	const blockSize = 4
	c := NewMemoryAwarePrefixCache(blockSize, 1000, 2500) // budget ~2 blocks

	first := seq(blockSize) // one block
	_, _ = c.Insert(first)
	_, handles := c.Match(first)
	c.Pin(handles)

	// Insert several more blocks; the pinned first block must survive.
	c.Insert(seq(blockSize * 5))

	if _, h := c.Match(first); len(h) != 1 {
		t.Fatal("pinned block was evicted despite a live reader")
	}
	c.Unpin(handles)
}

func TestNonPositiveBudgetDisablesTracking(t *testing.T) {
	c := NewMemoryAwarePrefixCache(4, 1000, 0)
	c.Insert(seq(4 * 10))
	if c.Len() != 10 {
		t.Fatalf("a disabled budget should keep every block, got %d", c.Len())
	}
	if c.UsageBytes() != 0 {
		t.Fatalf("untracked cache should report 0 usage, got %d", c.UsageBytes())
	}
}
