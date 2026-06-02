// SPDX-License-Identifier: Apache-2.0

package compute

import "testing"

func TestKVCacheReserveGrowsByStep(t *testing.T) {
	c := KVCache{Step: 4}
	// First prompt of 3 tokens: capacity rounds up to one step.
	start, end, capacity := c.Reserve(3)
	if start != 0 || end != 3 || capacity != 4 {
		t.Fatalf("first reserve: start=%d end=%d cap=%d", start, end, capacity)
	}
	// Two more tokens cross the step boundary: capacity grows to 8.
	start, end, capacity = c.Reserve(2)
	if start != 3 || end != 5 || capacity != 8 {
		t.Fatalf("second reserve: start=%d end=%d cap=%d", start, end, capacity)
	}
	if c.Offset != 5 {
		t.Errorf("offset: got %d want 5", c.Offset)
	}
}

func TestKVCacheDefaultStep(t *testing.T) {
	c := KVCache{}
	_, _, capacity := c.Reserve(1)
	if capacity != DefaultStep {
		t.Errorf("default step capacity: got %d want %d", capacity, DefaultStep)
	}
}

func TestKVCacheTrim(t *testing.T) {
	c := KVCache{Step: 8}
	c.Reserve(10)
	if got := c.Trim(3); got != 3 || c.Offset != 7 {
		t.Fatalf("trim 3: removed=%d offset=%d", got, c.Offset)
	}
	// Trimming past the start clamps.
	if got := c.Trim(100); got != 7 || c.Offset != 0 {
		t.Fatalf("trim past start: removed=%d offset=%d", got, c.Offset)
	}
}

func TestRotatingCacheFillsThenRings(t *testing.T) {
	c := RotatingKVCache{MaxSize: 4, Keep: 1}
	// Fill the window: slots 0,1,2,3.
	for i := 0; i < 4; i++ {
		if slot := c.Update(); slot != i {
			t.Fatalf("fill slot %d: got %d", i, slot)
		}
	}
	if c.Len() != 4 {
		t.Fatalf("len after fill: got %d want 4", c.Len())
	}
	// Now full. Keep=1 pins slot 0; the ring cycles slots 1,2,3.
	want := []int{1, 2, 3, 1, 2, 3}
	for i, w := range want {
		if slot := c.Update(); slot != w {
			t.Fatalf("ring step %d: got %d want %d", i, slot, w)
		}
	}
	if c.Len() != 4 {
		t.Errorf("len stays at window size: got %d", c.Len())
	}
}

func TestRotatingCacheNoKeep(t *testing.T) {
	c := RotatingKVCache{MaxSize: 2}
	c.Update() // slot 0
	c.Update() // slot 1
	// Full, nothing pinned: ring over both slots.
	want := []int{0, 1, 0, 1}
	for i, w := range want {
		if slot := c.Update(); slot != w {
			t.Fatalf("step %d: got %d want %d", i, slot, w)
		}
	}
}

func TestRotatingCacheUnbounded(t *testing.T) {
	c := RotatingKVCache{} // MaxSize 0 -> plain append
	for i := 0; i < 5; i++ {
		if slot := c.Update(); slot != i {
			t.Fatalf("unbounded slot %d: got %d", i, slot)
		}
	}
	if c.Len() != 5 {
		t.Errorf("unbounded len: got %d want 5", c.Len())
	}
}

func TestRotatingCacheReset(t *testing.T) {
	c := RotatingKVCache{MaxSize: 3, Keep: 1}
	c.Update()
	c.Update()
	c.Reset()
	if c.Len() != 0 || c.Offset != 0 {
		t.Errorf("reset: len=%d offset=%d", c.Len(), c.Offset)
	}
	if slot := c.Update(); slot != 0 {
		t.Errorf("first slot after reset: got %d want 0", slot)
	}
}

func TestRoundUp(t *testing.T) {
	cases := []struct{ n, step, want int }{
		{0, 4, 0}, {1, 4, 4}, {4, 4, 4}, {5, 4, 8}, {7, 0, 7},
	}
	for _, tc := range cases {
		if got := roundUp(tc.n, tc.step); got != tc.want {
			t.Errorf("roundUp(%d,%d): got %d want %d", tc.n, tc.step, got, tc.want)
		}
	}
}
