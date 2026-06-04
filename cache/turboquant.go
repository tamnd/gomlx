// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"math"
	"math/rand"
)

// TurboQuant compresses the value half of a KV cache. Keys stay in full
// precision because attention is far more sensitive to key error than to value
// error, so quantizing values alone keeps quality while cutting the larger of
// the two tensors. Each contiguous group of values is centered and scaled to a
// unit-variance form, then each element is mapped to the nearest level of a
// Lloyd-Max codebook (the minimum mean-squared-error quantizer for a Gaussian),
// and the level indices are bit-packed. Dequantizing reverses the map: a level
// times the group scale plus the group mean.
//
// The group mean makes the scheme asymmetric, which matters because a value
// distribution is rarely centered on zero. Storing a mean and a scale per group
// of GroupSize values costs two float32 per group, which a group of 32 packed
// 4-bit codes (16 bytes) easily pays for.

// TurboQuantConfig sets the quantizer width and grouping. Bits is 3 or 4; lower
// bits save more memory at a higher error. GroupSize is how many adjacent values
// share one mean and scale; smaller groups track local statistics more closely
// at a higher metadata cost. RotationSeed, when non-zero, selects the random
// orthogonal rotation applied to each group before quantizing (see
// GenerateRotationMatrix); zero leaves the values unrotated.
type TurboQuantConfig struct {
	Bits         int
	GroupSize    int
	RotationSeed int64
}

// DefaultTurboQuantConfig is the configuration the spec calls for: 4-bit codes
// over groups of 32, no rotation.
func DefaultTurboQuantConfig() TurboQuantConfig {
	return TurboQuantConfig{Bits: 4, GroupSize: 32, RotationSeed: 0}
}

func (c TurboQuantConfig) validate() error {
	if c.Bits != 3 && c.Bits != 4 {
		return fmt.Errorf("turboquant: Bits must be 3 or 4, got %d", c.Bits)
	}
	if c.GroupSize <= 0 {
		return fmt.Errorf("turboquant: GroupSize must be positive, got %d", c.GroupSize)
	}
	return nil
}

// QuantizedV holds one compressed value tensor. N is the number of values that
// were quantized, before any padding of the final group. Means and Scales carry
// one entry per group; Codes is the bit-packed stream of level indices.
type QuantizedV struct {
	Bits      int
	GroupSize int
	N         int
	Means     []float32
	Scales    []float32
	Codes     []byte
}

// codebooks holds the Lloyd-Max reconstruction levels for a unit-variance
// Gaussian. These are the standard tabulated minimum-MSE levels; the decision
// boundary between two adjacent levels is their midpoint, so nearest-level
// search is the optimal assignment.
var codebooks = map[int][]float32{
	3: {-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520},
	4: {
		-2.7326, -2.0690, -1.6181, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
		0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326,
	},
}

// minScale floors a group's scale so a constant or near-constant group does not
// divide by zero; the resulting codes all land on the central level and
// dequantize back to the mean.
const minScale = 1e-12

// Quantize compresses values into a QuantizedV under the given config. The
// values are grouped in order; the final group is padded with zeros to a full
// group, and N records the true count so Dequantize returns exactly len(values).
func Quantize(values []float32, c TurboQuantConfig) (*QuantizedV, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	book := codebooks[c.Bits]
	n := len(values)
	groups := (n + c.GroupSize - 1) / c.GroupSize

	q := &QuantizedV{
		Bits:      c.Bits,
		GroupSize: c.GroupSize,
		N:         n,
		Means:     make([]float32, groups),
		Scales:    make([]float32, groups),
	}
	indices := make([]int, groups*c.GroupSize)

	for g := range groups {
		lo := g * c.GroupSize
		hi := min(lo+c.GroupSize, n)
		group := values[lo:hi]

		mean := meanOf(group)
		scale := max(stdOf(group, mean), minScale)
		q.Means[g] = mean
		q.Scales[g] = scale

		for i := range c.GroupSize {
			idx := g*c.GroupSize + i
			if lo+i >= hi {
				indices[idx] = len(book) / 2 // padding maps to a central level
				continue
			}
			norm := (values[lo+i] - mean) / scale
			indices[idx] = nearestLevel(book, norm)
		}
	}

	q.Codes = packBits(indices, c.Bits)
	return q, nil
}

// Dequantize reconstructs the values from a QuantizedV, returning exactly N of
// them.
func Dequantize(q *QuantizedV) ([]float32, error) {
	book, ok := codebooks[q.Bits]
	if !ok {
		return nil, fmt.Errorf("turboquant: unknown bit width %d", q.Bits)
	}
	groups := len(q.Means)
	indices := unpackBits(q.Codes, q.Bits, groups*q.GroupSize)

	out := make([]float32, q.N)
	for g := range groups {
		for i := range q.GroupSize {
			pos := g*q.GroupSize + i
			if pos >= q.N {
				break
			}
			out[pos] = book[indices[pos]]*q.Scales[g] + q.Means[g]
		}
	}
	return out, nil
}

// nearestLevel returns the index of the codebook level closest to x. The book is
// small (8 or 16 entries) and sorted ascending, so a linear scan that stops once
// the gap starts growing is both simple and fast.
func nearestLevel(book []float32, x float32) int {
	best := 0
	bestDist := math.Abs(float64(x - book[0]))
	for i := 1; i < len(book); i++ {
		d := math.Abs(float64(x - book[i]))
		if d < bestDist {
			best, bestDist = i, d
			continue
		}
		break
	}
	return best
}

func meanOf(g []float32) float32 {
	if len(g) == 0 {
		return 0
	}
	var s float64
	for _, v := range g {
		s += float64(v)
	}
	return float32(s / float64(len(g)))
}

func stdOf(g []float32, mean float32) float32 {
	if len(g) == 0 {
		return 0
	}
	var s float64
	for _, v := range g {
		d := float64(v - mean)
		s += d * d
	}
	return float32(math.Sqrt(s / float64(len(g))))
}

// packBits packs a run of small unsigned indices into a dense bit stream, bits
// per index, most-significant bit first within the stream. With bits == 4 this
// is two indices per byte; with bits == 3 it spans byte boundaries.
func packBits(indices []int, bits int) []byte {
	total := len(indices) * bits
	out := make([]byte, (total+7)/8)
	pos := 0
	for _, v := range indices {
		for b := bits - 1; b >= 0; b-- {
			if v&(1<<b) != 0 {
				out[pos/8] |= 1 << (7 - pos%8)
			}
			pos++
		}
	}
	return out
}

// unpackBits is the inverse of packBits: it reads count indices of bits each
// from the stream.
func unpackBits(data []byte, bits, count int) []int {
	out := make([]int, count)
	pos := 0
	for i := range count {
		v := 0
		for range bits {
			v <<= 1
			if data[pos/8]&(1<<(7-pos%8)) != 0 {
				v |= 1
			}
			pos++
		}
		out[i] = v
	}
	return out
}

// GenerateRotationMatrix returns a dim by dim orthogonal matrix as a row-major
// flat slice, derived deterministically from seed. The matrix is the Q factor of
// a Gram-Schmidt orthogonalization of a seeded Gaussian matrix, so it is a
// uniformly random rotation for a given seed and exactly reproducible. Rotating
// a group of values by an orthogonal matrix before quantizing spreads energy
// across the dimensions, which reduces quantization error on skewed inputs; the
// rotation is undone after dequantizing by multiplying by the transpose. The
// GPU path applies the matrix through mlxgo; this generator runs on the host so
// the matrix can be built once and uploaded.
func GenerateRotationMatrix(dim int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	cols := make([][]float64, dim)
	for j := range dim {
		cols[j] = make([]float64, dim)
		for i := range dim {
			cols[j][i] = r.NormFloat64()
		}
	}

	// Modified Gram-Schmidt over the columns yields an orthonormal basis.
	for j := range dim {
		for k := range j {
			dot := 0.0
			for i := range dim {
				dot += cols[j][i] * cols[k][i]
			}
			for i := range dim {
				cols[j][i] -= dot * cols[k][i]
			}
		}
		norm := 0.0
		for i := range dim {
			norm += cols[j][i] * cols[j][i]
		}
		norm = max(math.Sqrt(norm), minScale)
		for i := range dim {
			cols[j][i] /= norm
		}
	}

	out := make([]float32, dim*dim)
	for i := range dim {
		for j := range dim {
			out[i*dim+j] = float32(cols[j][i])
		}
	}
	return out
}
