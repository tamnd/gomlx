// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"strconv"
	"sync/atomic"
)

// idGen produces unique response IDs of the form "<prefix>-<counter>" without a
// time or randomness dependency, which keeps the hot path allocation-light and
// tests deterministic across a process.
type idGen struct {
	prefix string
	seq    atomic.Uint64
}

func newIDGen(prefix string) *idGen { return &idGen{prefix: prefix + "-"} }

func (g *idGen) next() string {
	n := g.seq.Add(1)
	return g.prefix + strconv.FormatUint(n, 36)
}
