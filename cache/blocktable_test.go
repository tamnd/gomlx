// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"errors"
	"testing"
)

// TestAppendGrowsByWholeBlocks checks the table allocates a new physical block
// only when the tail block fills.
func TestAppendGrowsByWholeBlocks(t *testing.T) {
	pool := NewBlockPool(4, 8)
	tab := NewBlockTable(pool)

	added, err := tab.Append(3) // first block
	if err != nil || len(added) != 1 {
		t.Fatalf("append 3: added=%v err=%v", added, err)
	}
	if tab.Len() != 3 || len(tab.Blocks()) != 1 {
		t.Fatalf("after 3: len=%d blocks=%d", tab.Len(), len(tab.Blocks()))
	}

	added, _ = tab.Append(1) // fills the first block, no new block
	if len(added) != 0 {
		t.Fatalf("append filling block should add none, got %v", added)
	}
	added, _ = tab.Append(1) // spills into a second block
	if len(added) != 1 || len(tab.Blocks()) != 2 {
		t.Fatalf("append spill: added=%v blocks=%d", added, len(tab.Blocks()))
	}
	if tab.Len() != 5 {
		t.Fatalf("len after spill: %d want 5", tab.Len())
	}
}

// TestForkSharesPrefixBlocks checks that forking a cached prefix shares blocks
// rather than allocating, and that appends after the fork allocate fresh blocks.
func TestForkSharesPrefixBlocks(t *testing.T) {
	pool := NewBlockPool(4, 8)
	// Stand in for a cached prefix already resident in the pool.
	prefix, _ := pool.AllocateN(2)
	freeBefore := pool.FreeCount()

	tab := NewBlockTable(pool)
	tab.Fork(prefix)
	if tab.SharedBlocks() != 2 || tab.Len() != 8 {
		t.Fatalf("after fork: shared=%d len=%d", tab.SharedBlocks(), tab.Len())
	}
	if pool.FreeCount() != freeBefore {
		t.Fatalf("fork allocated new blocks: free went %d -> %d", freeBefore, pool.FreeCount())
	}
	for _, id := range prefix {
		if pool.RefCount(id) != 2 {
			t.Fatalf("forked block %d ref=%d want 2", id, pool.RefCount(id))
		}
	}

	added, err := tab.Append(3) // beyond the shared prefix
	if err != nil || len(added) != 1 {
		t.Fatalf("append after fork: added=%v err=%v", added, err)
	}
	if len(tab.Blocks()) != 3 {
		t.Fatalf("blocks after fork+append: %d want 3", len(tab.Blocks()))
	}
}

// TestFreeReleasesBlocks checks that freeing a table returns its blocks and
// only drops a shared block's last reference.
func TestFreeReleasesBlocks(t *testing.T) {
	pool := NewBlockPool(4, 8)
	prefix, _ := pool.AllocateN(1)

	tab := NewBlockTable(pool)
	tab.Fork(prefix)
	tab.Append(5) // adds blocks of its own
	tab.Free()

	if tab.Len() != 0 || len(tab.Blocks()) != 0 {
		t.Fatalf("table not cleared after free: len=%d blocks=%d", tab.Len(), len(tab.Blocks()))
	}
	// The shared prefix block still has the original allocation's reference.
	if pool.RefCount(prefix[0]) != 1 {
		t.Fatalf("shared block ref after free: %d want 1", pool.RefCount(prefix[0]))
	}
}

// TestAppendOutOfBlocksLeavesTableUnchanged checks that a failed append does not
// half-grow the table.
func TestAppendOutOfBlocksLeavesTableUnchanged(t *testing.T) {
	pool := NewBlockPool(4, 1)
	tab := NewBlockTable(pool)
	if _, err := tab.Append(3); err != nil { // uses the one block
		t.Fatalf("first append: %v", err)
	}
	_, err := tab.Append(10) // needs more blocks than the pool has
	if !errors.Is(err, ErrOutOfBlocks) {
		t.Fatalf("expected ErrOutOfBlocks, got %v", err)
	}
	if tab.Len() != 3 || len(tab.Blocks()) != 1 {
		t.Fatalf("failed append mutated table: len=%d blocks=%d", tab.Len(), len(tab.Blocks()))
	}
}
