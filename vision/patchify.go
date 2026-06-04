// SPDX-License-Identifier: Apache-2.0

package vision

import "fmt"

// A vision transformer does not read pixels directly: it cuts the image into a
// grid of fixed-size square patches, flattens each patch into a vector, and
// projects that vector to the model width. The projection is a matrix multiply a
// GPU runs, but cutting the image into the flattened patch sequence is a pure
// reshape of the pixel tensor, so it belongs on the host alongside preprocessing.
// Patchify produces exactly the sequence that patch projection consumes.

// PatchGrid is an image cut into a sequence of flattened patches. Data is laid
// out as NumPatches rows of PatchDim values; patches run across the grid row by
// row, and within a patch the values run channel-major (all of channel 0's
// patch pixels, then channel 1, then channel 2), matching how a convolutional
// patch-embedding weight reshapes to a matrix.
type PatchGrid struct {
	NumPatches int
	PatchDim   int // channels * patch * patch
	GridH      int
	GridW      int
	Data       []float32
}

// Patch returns the flattened vector for one patch by its index in the sequence.
// It is a convenience for callers and tests that read a single patch without
// recomputing the stride.
func (g *PatchGrid) Patch(i int) []float32 {
	return g.Data[i*g.PatchDim : (i+1)*g.PatchDim]
}

// Patchify cuts CHW pixel values into a grid of patch by patch squares and
// flattens each into a row. The image dimensions must be exact multiples of the
// patch size, which is how vision towers are configured; a non-multiple is an
// error rather than a silent crop so a misconfigured size surfaces at the call.
func Patchify(pv *PixelValues, patch int) (*PatchGrid, error) {
	if patch <= 0 {
		return nil, fmt.Errorf("vision: patch size must be positive, got %d", patch)
	}
	if pv.Height%patch != 0 || pv.Width%patch != 0 {
		return nil, fmt.Errorf("vision: image %dx%d is not a multiple of patch size %d",
			pv.Width, pv.Height, patch)
	}

	gridH := pv.Height / patch
	gridW := pv.Width / patch
	numPatches := gridH * gridW
	patchDim := pv.Channels * patch * patch
	plane := pv.Height * pv.Width

	g := &PatchGrid{
		NumPatches: numPatches,
		PatchDim:   patchDim,
		GridH:      gridH,
		GridW:      gridW,
		Data:       make([]float32, numPatches*patchDim),
	}

	for gy := range gridH {
		for gx := range gridW {
			patchIdx := gy*gridW + gx
			base := patchIdx * patchDim
			y0 := gy * patch
			x0 := gx * patch
			// Channel-major within the patch: channel 0's whole patch, then 1,
			// then 2, each scanned row by row.
			for c := range pv.Channels {
				cbase := c * plane
				for py := range patch {
					row := cbase + (y0+py)*pv.Width + x0
					dst := base + (c*patch+py)*patch
					copy(g.Data[dst:dst+patch], pv.Data[row:row+patch])
				}
			}
		}
	}
	return g, nil
}
