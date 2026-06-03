// SPDX-License-Identifier: Apache-2.0

package cache

// BlockTable is one sequence's map from logical token positions to the physical
// key/value blocks that hold them. As a sequence decodes it grows by whole
// blocks: the table appends a fresh physical block from the pool whenever the
// last block fills. When a sequence starts on a cached prefix it forks the
// shared blocks instead of recomputing them, which raises their reference count
// in the pool rather than copying any state.
//
// A BlockTable is owned by a single sequence on the scheduler goroutine and is
// not safe for concurrent use.
type BlockTable struct {
	pool   *BlockPool
	blocks []int // physical block ids in logical order
	shared int   // count of leading blocks forked from a shared prefix
	tokens int   // logical tokens recorded so far
}

// NewBlockTable returns an empty table backed by pool.
func NewBlockTable(pool *BlockPool) *BlockTable {
	return &BlockTable{pool: pool}
}

// Len reports how many logical tokens the table currently covers.
func (t *BlockTable) Len() int { return t.tokens }

// Blocks returns the physical block ids in logical order. The slice is the
// table's own backing array; callers must not mutate it.
func (t *BlockTable) Blocks() []int { return t.blocks }

// SharedBlocks reports how many leading blocks were forked from a shared prefix
// and so were not recomputed.
func (t *BlockTable) SharedBlocks() int { return t.shared }

// Fork seeds the table with blocks shared from a cached prefix, raising each
// block's reference count in the pool so it is not evicted while this sequence
// reads it. It must be called on an empty table, before any Append, and the
// prefix is taken to be exactly len(shared) whole blocks of cached tokens.
func (t *BlockTable) Fork(shared []int) {
	for _, id := range shared {
		t.pool.Share(id)
		t.blocks = append(t.blocks, id)
	}
	t.shared = len(shared)
	t.tokens = len(shared) * t.pool.BlockSize()
}

// Append grows the table to cover n additional tokens, allocating physical
// blocks from the pool as the tail block fills. It returns the ids of any blocks
// newly allocated by this call, in order, so the caller knows which blocks'
// device state it must compute. If the pool cannot supply a needed block it
// allocates none for the shortfall and returns ErrOutOfBlocks, leaving the table
// unchanged from before the call so the scheduler can retry after freeing space.
func (t *BlockTable) Append(n int) (added []int, err error) {
	if n <= 0 {
		return nil, nil
	}
	need := t.pool.BlocksFor(t.tokens+n) - len(t.blocks)
	if need > 0 {
		ids, allocErr := t.pool.AllocateN(need)
		if allocErr != nil {
			return nil, allocErr
		}
		t.blocks = append(t.blocks, ids...)
		added = ids
	}
	t.tokens += n
	return added, nil
}

// Free releases every block the table holds back to the pool and empties the
// table. Shared blocks are reference counted, so this only returns a forked
// prefix block to the free list when the last sequence using it frees it.
func (t *BlockTable) Free() {
	t.pool.FreeAll(t.blocks)
	t.blocks = nil
	t.shared = 0
	t.tokens = 0
}
