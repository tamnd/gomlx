// SPDX-License-Identifier: Apache-2.0

package cache

import "errors"

// ErrOutOfBlocks is returned when the pool has no free physical block to hand
// out. The scheduler treats this as backpressure: it stops admitting new work
// until running sequences free blocks.
var ErrOutOfBlocks = errors.New("cache: out of physical blocks")

// BlockPool manages a fixed set of physical key/value blocks, each holding one
// block's worth of token slots. Paged attention stores a sequence's keys and
// values in these blocks rather than one contiguous span, so the memory for a
// long context does not have to be reserved up front and blocks can be shared
// between sequences that share a prefix.
//
// Each block carries a reference count. A block handed to one sequence has a
// count of one; sharing it with another sequence (a cache hit on a common
// prefix) raises the count, and freeing it lowers the count. The block returns
// to the free list only when the last reference is freed. BlockPool is not safe
// for concurrent use; the engine drives it from the scheduler goroutine.
type BlockPool struct {
	blockSize int
	total     int
	free      []int // stack of free physical block ids
	refs      []int // reference count per physical block id
}

// NewBlockPool creates a pool of total physical blocks, each sized to hold
// blockSize token positions of key/value state.
func NewBlockPool(blockSize, total int) *BlockPool {
	if blockSize <= 0 {
		blockSize = 16
	}
	if total < 0 {
		total = 0
	}
	p := &BlockPool{
		blockSize: blockSize,
		total:     total,
		free:      make([]int, total),
		refs:      make([]int, total),
	}
	// Hand out low ids first by keeping the highest id on top of the stack.
	for i := 0; i < total; i++ {
		p.free[i] = total - 1 - i
	}
	return p
}

// BlockSize reports the token capacity of one physical block.
func (p *BlockPool) BlockSize() int { return p.blockSize }

// Total reports the number of physical blocks the pool manages.
func (p *BlockPool) Total() int { return p.total }

// FreeCount reports how many physical blocks are currently unallocated.
func (p *BlockPool) FreeCount() int { return len(p.free) }

// BlocksFor reports how many physical blocks a sequence of n tokens needs, which
// is n divided by the block size rounded up.
func (p *BlockPool) BlocksFor(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + p.blockSize - 1) / p.blockSize
}

// Allocate takes one physical block from the free list and returns its id with a
// reference count of one. It returns ErrOutOfBlocks when the pool is exhausted.
func (p *BlockPool) Allocate() (int, error) {
	n := len(p.free)
	if n == 0 {
		return 0, ErrOutOfBlocks
	}
	id := p.free[n-1]
	p.free = p.free[:n-1]
	p.refs[id] = 1
	return id, nil
}

// AllocateN takes count physical blocks at once. If the pool cannot satisfy the
// whole request it allocates none and returns ErrOutOfBlocks, so a sequence
// never ends up with a partial block table it cannot use.
func (p *BlockPool) AllocateN(count int) ([]int, error) {
	if count <= 0 {
		return nil, nil
	}
	if count > len(p.free) {
		return nil, ErrOutOfBlocks
	}
	ids := make([]int, count)
	for i := range ids {
		id, _ := p.Allocate()
		ids[i] = id
	}
	return ids, nil
}

// Share raises the reference count of an already-allocated block, recording that
// another sequence now reads it. This is how a prefix-cache hit avoids copying:
// the new sequence points its block table at the existing physical block.
func (p *BlockPool) Share(id int) {
	if id >= 0 && id < p.total && p.refs[id] > 0 {
		p.refs[id]++
	}
}

// Free drops one reference to a block. When the last reference goes away the
// block returns to the free list. Freeing an already-free block is a no-op.
func (p *BlockPool) Free(id int) {
	if id < 0 || id >= p.total || p.refs[id] == 0 {
		return
	}
	p.refs[id]--
	if p.refs[id] == 0 {
		p.free = append(p.free, id)
	}
}

// FreeAll frees a slice of blocks, the usual case when a sequence finishes.
func (p *BlockPool) FreeAll(ids []int) {
	for _, id := range ids {
		p.Free(id)
	}
}

// RefCount reports the current reference count of a block, mainly for tests and
// telemetry.
func (p *BlockPool) RefCount(id int) int {
	if id < 0 || id >= p.total {
		return 0
	}
	return p.refs[id]
}
