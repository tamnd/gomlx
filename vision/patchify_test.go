// SPDX-License-Identifier: Apache-2.0

package vision

import "testing"

// ramp builds CHW pixel values whose value at (c,y,x) encodes its coordinates, so
// a test can assert exactly which pixel landed in which patch slot.
func ramp(channels, h, w int) *PixelValues {
	pv := &PixelValues{Channels: channels, Height: h, Width: w,
		Data: make([]float32, channels*h*w)}
	for c := range channels {
		for y := range h {
			for x := range w {
				pv.Data[(c*h+y)*w+x] = float32(c*10000 + y*100 + x)
			}
		}
	}
	return pv
}

func TestPatchifyShape(t *testing.T) {
	pv := ramp(3, 8, 8)
	g, err := Patchify(pv, 4)
	if err != nil {
		t.Fatalf("patchify: %v", err)
	}
	if g.GridH != 2 || g.GridW != 2 || g.NumPatches != 4 {
		t.Fatalf("grid: %dx%d numPatches=%d", g.GridH, g.GridW, g.NumPatches)
	}
	if g.PatchDim != 3*4*4 {
		t.Fatalf("patchDim: got %d want %d", g.PatchDim, 3*4*4)
	}
	if len(g.Data) != 4*3*4*4 {
		t.Fatalf("data length: got %d want %d", len(g.Data), 4*3*4*4)
	}
}

func TestPatchifyLayout(t *testing.T) {
	// 4x4 image, patch 2: four patches in row-major grid order. Check the corners
	// of each patch and the channel-major flattening within a patch.
	pv := ramp(3, 4, 4)
	g, err := Patchify(pv, 2)
	if err != nil {
		t.Fatalf("patchify: %v", err)
	}

	// Patch 0 is the top-left 2x2 block: rows 0..1, cols 0..1.
	p0 := g.Patch(0)
	// Channel 0, in-patch (py,px) row-major: (0,0)=0, (0,1)=1, (1,0)=100, (1,1)=101.
	wantC0 := []float32{0, 1, 100, 101}
	for i, w := range wantC0 {
		if p0[i] != w {
			t.Errorf("patch0 channel0 [%d]: got %v want %v", i, p0[i], w)
		}
	}
	// Channel 1 of patch 0 begins after the 4 channel-0 values.
	if p0[4] != 10000 {
		t.Errorf("patch0 channel1 first value: got %v want 10000", p0[4])
	}
	// Channel 2 begins after 8 values.
	if p0[8] != 20000 {
		t.Errorf("patch0 channel2 first value: got %v want 20000", p0[8])
	}

	// Patch 1 is the top-right block: cols 2..3. Its channel-0 first value is x=2.
	p1 := g.Patch(1)
	if p1[0] != 2 {
		t.Errorf("patch1 channel0 first value: got %v want 2", p1[0])
	}
	// Patch 2 is bottom-left: rows 2..3. First value y=2 -> 200.
	p2 := g.Patch(2)
	if p2[0] != 200 {
		t.Errorf("patch2 channel0 first value: got %v want 200", p2[0])
	}
	// Patch 3 is bottom-right: row 2 col 2 -> 202.
	p3 := g.Patch(3)
	if p3[0] != 202 {
		t.Errorf("patch3 channel0 first value: got %v want 202", p3[0])
	}
}

func TestPatchifyRejectsNonMultiple(t *testing.T) {
	pv := ramp(3, 10, 8)
	if _, err := Patchify(pv, 4); err == nil {
		t.Error("a height that is not a multiple of the patch size must error")
	}
	if _, err := Patchify(ramp(3, 8, 8), 0); err == nil {
		t.Error("a zero patch size must error")
	}
}

func TestPatchifyRoundTripsAllPixels(t *testing.T) {
	// Every pixel must appear exactly once across all patches: the total count of
	// values equals the pixel tensor length, and reassembling recovers each one.
	pv := ramp(3, 12, 12)
	g, err := Patchify(pv, 3)
	if err != nil {
		t.Fatalf("patchify: %v", err)
	}
	if len(g.Data) != len(pv.Data) {
		t.Fatalf("total values: got %d want %d", len(g.Data), len(pv.Data))
	}

	seen := make(map[float32]int)
	for _, v := range g.Data {
		seen[v]++
	}
	for _, v := range pv.Data {
		if seen[v] != 1 {
			t.Fatalf("pixel value %v appears %d times, want once", v, seen[v])
		}
	}
}
