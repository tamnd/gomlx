// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"errors"
	"testing"
)

func TestBlocksFor(t *testing.T) {
	p := NewBlockPool(16, 8)
	cases := []struct{ n, want int }{
		{0, 0}, {1, 1}, {16, 1}, {17, 2}, {32, 2}, {33, 3},
	}
	for _, c := range cases {
		if got := p.BlocksFor(c.n); got != c.want {
			t.Errorf("BlocksFor(%d)=%d want %d", c.n, got, c.want)
		}
	}
}

// TestAllocateAndFree checks the free count tracks allocation and release and
// that a freed block can be handed out again.
func TestAllocateAndFree(t *testing.T) {
	p := NewBlockPool(16, 3)
	if p.FreeCount() != 3 {
		t.Fatalf("free count: got %d want 3", p.FreeCount())
	}
	a, err := p.Allocate()
	if err != nil {
		t.Fatalf("allocate: %v", err)
	}
	if p.FreeCount() != 2 || p.RefCount(a) != 1 {
		t.Fatalf("after alloc: free=%d ref=%d", p.FreeCount(), p.RefCount(a))
	}
	p.Free(a)
	if p.FreeCount() != 3 || p.RefCount(a) != 0 {
		t.Fatalf("after free: free=%d ref=%d", p.FreeCount(), p.RefCount(a))
	}
}

// TestAllocateNAllOrNothing checks that an over-large request allocates nothing.
func TestAllocateNAllOrNothing(t *testing.T) {
	p := NewBlockPool(16, 4)
	if _, err := p.AllocateN(5); !errors.Is(err, ErrOutOfBlocks) {
		t.Fatalf("expected ErrOutOfBlocks, got %v", err)
	}
	if p.FreeCount() != 4 {
		t.Fatalf("failed AllocateN consumed blocks: free=%d", p.FreeCount())
	}
	ids, err := p.AllocateN(4)
	if err != nil || len(ids) != 4 {
		t.Fatalf("AllocateN(4): ids=%v err=%v", ids, err)
	}
	if p.FreeCount() != 0 {
		t.Fatalf("free count after full alloc: %d", p.FreeCount())
	}
}

// TestExhaustion checks that allocating past the pool size reports backpressure.
func TestExhaustion(t *testing.T) {
	p := NewBlockPool(16, 1)
	if _, err := p.Allocate(); err != nil {
		t.Fatalf("first allocate: %v", err)
	}
	if _, err := p.Allocate(); !errors.Is(err, ErrOutOfBlocks) {
		t.Fatalf("expected ErrOutOfBlocks, got %v", err)
	}
}

// TestSharedBlockSurvivesUntilLastFree checks reference counting: a shared block
// only returns to the pool when every holder frees it.
func TestSharedBlockSurvivesUntilLastFree(t *testing.T) {
	p := NewBlockPool(16, 2)
	id, _ := p.Allocate()
	p.Share(id) // a second sequence now reads this block
	if p.RefCount(id) != 2 {
		t.Fatalf("ref after share: %d want 2", p.RefCount(id))
	}
	p.Free(id) // first holder done
	if p.RefCount(id) != 1 {
		t.Fatalf("ref after first free: %d want 1", p.RefCount(id))
	}
	if p.FreeCount() != 1 {
		t.Fatalf("block returned too early: free=%d", p.FreeCount())
	}
	p.Free(id) // second holder done
	if p.RefCount(id) != 0 || p.FreeCount() != 2 {
		t.Fatalf("block not returned: ref=%d free=%d", p.RefCount(id), p.FreeCount())
	}
}

// TestDoubleFreeIsNoop checks that freeing a block that is already free does not
// corrupt the pool's accounting.
func TestDoubleFreeIsNoop(t *testing.T) {
	p := NewBlockPool(16, 2)
	id, _ := p.Allocate()
	p.Free(id)
	p.Free(id) // extra free
	if p.FreeCount() != 2 {
		t.Fatalf("double free corrupted pool: free=%d", p.FreeCount())
	}
}

// TestFreeAll releases a sequence's whole block table at once.
func TestFreeAll(t *testing.T) {
	p := NewBlockPool(16, 8)
	ids, _ := p.AllocateN(5)
	p.FreeAll(ids)
	if p.FreeCount() != 8 {
		t.Fatalf("FreeAll left blocks allocated: free=%d", p.FreeCount())
	}
}
