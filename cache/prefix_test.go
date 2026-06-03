// SPDX-License-Identifier: Apache-2.0

package cache

import "testing"

func seq(n int) []int32 {
	t := make([]int32, n)
	for i := range t {
		t[i] = int32(i + 1)
	}
	return t
}

// TestInsertThenMatchWholePrefix inserts a prompt and checks that an identical
// prompt matches every whole block.
func TestInsertThenMatchWholePrefix(t *testing.T) {
	c := NewPrefixCache(4, 0)
	tokens := seq(12) // three whole blocks
	handles, added := c.Insert(tokens)
	if len(handles) != 3 || len(added) != 3 {
		t.Fatalf("insert: got %d handles, %d added", len(handles), len(added))
	}
	for i, a := range added {
		if !a {
			t.Errorf("block %d should be new on first insert", i)
		}
	}

	matched, got := c.Match(tokens)
	if matched != 12 {
		t.Fatalf("matched %d tokens, want 12", matched)
	}
	for i := range got {
		if got[i] != handles[i] {
			t.Errorf("block %d handle: got %d want %d", i, got[i], handles[i])
		}
	}
}

// TestMatchStopsAtDivergence checks that matching stops at the first block whose
// tokens differ, so the result is a true shared prefix.
func TestMatchStopsAtDivergence(t *testing.T) {
	c := NewPrefixCache(4, 0)
	c.Insert(seq(12))

	probe := seq(12)
	probe[9] = 999 // change a token in the third block
	matched, got := c.Match(probe)
	if matched != 8 {
		t.Fatalf("matched %d tokens, want 8 (first two blocks)", matched)
	}
	if len(got) != 2 {
		t.Fatalf("got %d handles, want 2", len(got))
	}
}

// TestTrailingPartialBlockIgnored checks that tokens past the last whole block
// are never matched, since only whole blocks carry cached state.
func TestTrailingPartialBlockIgnored(t *testing.T) {
	c := NewPrefixCache(4, 0)
	handles, _ := c.Insert(seq(10)) // two whole blocks, two leftover tokens
	if len(handles) != 2 {
		t.Fatalf("insert returned %d handles, want 2", len(handles))
	}
	matched, _ := c.Match(seq(10))
	if matched != 8 {
		t.Fatalf("matched %d tokens, want 8", matched)
	}
}

// TestPrefixDoesNotAlias checks that the same block of tokens reached through a
// different prefix hashes to a different block and does not collide.
func TestPrefixDoesNotAlias(t *testing.T) {
	c := NewPrefixCache(2, 0)
	a := []int32{1, 2, 7, 8}
	b := []int32{5, 6, 7, 8}
	ha, _ := c.Insert(a)
	hb, _ := c.Insert(b)
	// First blocks differ, so the second blocks ([7,8] under different parents)
	// must also be distinct handles.
	if ha[1] == hb[1] {
		t.Fatalf("second block aliased across prefixes: %d == %d", ha[1], hb[1])
	}
	if c.Len() != 4 {
		t.Fatalf("resident blocks: got %d, want 4", c.Len())
	}
}

// TestSecondInsertReusesBlocks checks that re-inserting an overlapping prompt
// reports the shared blocks as reused rather than new.
func TestSecondInsertReusesBlocks(t *testing.T) {
	c := NewPrefixCache(4, 0)
	c.Insert(seq(8)) // blocks A, B
	full := seq(12)  // blocks A, B, C
	_, added := c.Insert(full)
	want := []bool{false, false, true}
	if len(added) != 3 {
		t.Fatalf("added len %d, want 3", len(added))
	}
	for i := range want {
		if added[i] != want[i] {
			t.Errorf("added[%d]=%v want %v", i, added[i], want[i])
		}
	}
}

// TestEvictionDropsLeastRecentlyUsed fills the cache past capacity and checks
// the least-recently-used block is the one dropped.
func TestEvictionDropsLeastRecentlyUsed(t *testing.T) {
	c := NewPrefixCache(4, 2) // hold at most two blocks
	a := []int32{1, 2, 3, 4}
	b := []int32{10, 20, 30, 40}
	d := []int32{11, 21, 31, 41}
	c.Insert(a)
	c.Insert(b)
	// Touch a so b becomes least recently used.
	c.Match(a)
	c.Insert(d) // forces eviction of b
	if c.Len() != 2 {
		t.Fatalf("resident blocks: got %d, want 2", c.Len())
	}
	if m, _ := c.Match(a); m != 4 {
		t.Errorf("a should survive: matched %d", m)
	}
	if m, _ := c.Match(b); m != 0 {
		t.Errorf("b should be evicted: matched %d", m)
	}
}

// TestPinnedBlocksSurviveEviction checks that a pinned block is never evicted
// even when it is the least recently used.
func TestPinnedBlocksSurviveEviction(t *testing.T) {
	c := NewPrefixCache(4, 1)
	a := []int32{1, 2, 3, 4}
	ha, _ := c.Insert(a)
	c.Pin(ha)

	b := []int32{9, 8, 7, 6}
	c.Insert(b) // would evict a, but a is pinned
	if m, _ := c.Match(a); m != 4 {
		t.Errorf("pinned block a was evicted: matched %d", m)
	}

	c.Unpin(ha)
	d := []int32{5, 5, 5, 5}
	c.Insert(d)
	c.Insert(seq(4)) // push past capacity again now that a is unpinned
	// a is now evictable; the cache must stay within capacity.
	if c.Len() > 1 {
		t.Errorf("cache over capacity after unpin: %d", c.Len())
	}
}

// TestStatsCountHitsAndMisses checks the hit and miss counters.
func TestStatsCountHitsAndMisses(t *testing.T) {
	c := NewPrefixCache(4, 0)
	c.Insert(seq(8))
	c.Match(seq(8))              // hit
	c.Match([]int32{9, 9, 9, 9}) // miss
	hits, misses := c.Stats()
	if hits != 1 || misses != 1 {
		t.Fatalf("stats: hits=%d misses=%d, want 1 and 1", hits, misses)
	}
}
