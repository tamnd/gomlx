// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"math"
	"math/rand"
	"testing"
)

// gaussian builds n deterministic unit-variance samples for the round-trip
// tests.
func gaussian(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// normMSE is the mean squared error between two slices divided by the variance
// of the reference, so it is comparable across scales.
func normMSE(ref, got []float32) float64 {
	var mean float64
	for _, v := range ref {
		mean += float64(v)
	}
	mean /= float64(len(ref))

	var se, varr float64
	for i := range ref {
		d := float64(ref[i] - got[i])
		se += d * d
		c := float64(ref[i]) - mean
		varr += c * c
	}
	return se / varr
}

func TestQuantizeRoundTripErrorBound(t *testing.T) {
	values := gaussian(4096, 1)

	q4, err := Quantize(values, TurboQuantConfig{Bits: 4, GroupSize: 32})
	if err != nil {
		t.Fatalf("quantize 4-bit: %v", err)
	}
	got4, err := Dequantize(q4)
	if err != nil {
		t.Fatalf("dequantize 4-bit: %v", err)
	}
	mse4 := normMSE(values, got4)
	if mse4 > 0.03 {
		t.Errorf("4-bit normalized MSE %.4f exceeds 0.03", mse4)
	}

	q3, err := Quantize(values, TurboQuantConfig{Bits: 3, GroupSize: 32})
	if err != nil {
		t.Fatalf("quantize 3-bit: %v", err)
	}
	got3, err := Dequantize(q3)
	if err != nil {
		t.Fatalf("dequantize 3-bit: %v", err)
	}
	mse3 := normMSE(values, got3)
	if mse3 > 0.06 {
		t.Errorf("3-bit normalized MSE %.4f exceeds 0.06", mse3)
	}

	// More bits must mean less error on the same data.
	if mse4 >= mse3 {
		t.Errorf("4-bit error %.4f not below 3-bit error %.4f", mse4, mse3)
	}
}

func TestQuantizeLengthPreserved(t *testing.T) {
	// A length that is not a multiple of the group size exercises the padded
	// final group; Dequantize must still return exactly the input length.
	values := gaussian(100, 2)
	q, err := Quantize(values, TurboQuantConfig{Bits: 4, GroupSize: 32})
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}
	got, err := Dequantize(q)
	if err != nil {
		t.Fatalf("dequantize: %v", err)
	}
	if len(got) != len(values) {
		t.Fatalf("length: got %d want %d", len(got), len(values))
	}
}

func TestQuantizeConstantGroup(t *testing.T) {
	// A constant group has zero variance; the scale floor must keep it finite and
	// reconstruct the constant from the group mean.
	values := make([]float32, 64)
	for i := range values {
		values[i] = 3.5
	}
	q, err := Quantize(values, TurboQuantConfig{Bits: 4, GroupSize: 32})
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}
	got, err := Dequantize(q)
	if err != nil {
		t.Fatalf("dequantize: %v", err)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.Abs(float64(v-3.5)) > 1e-4 {
			t.Fatalf("constant group [%d]: got %v want 3.5", i, v)
		}
	}
}

func TestPackBitsRoundTrip(t *testing.T) {
	for _, bits := range []int{3, 4} {
		r := rand.New(rand.NewSource(int64(bits)))
		n := 1000
		idx := make([]int, n)
		for i := range idx {
			idx[i] = r.Intn(1 << bits)
		}
		packed := packBits(idx, bits)
		if want := (n*bits + 7) / 8; len(packed) != want {
			t.Errorf("bits=%d packed length: got %d want %d", bits, len(packed), want)
		}
		out := unpackBits(packed, bits, n)
		for i := range idx {
			if out[i] != idx[i] {
				t.Fatalf("bits=%d index %d: got %d want %d", bits, i, out[i], idx[i])
			}
		}
	}
}

func TestQuantizeConfigValidation(t *testing.T) {
	if _, err := Quantize([]float32{1, 2, 3}, TurboQuantConfig{Bits: 5, GroupSize: 32}); err == nil {
		t.Error("Bits=5 must be rejected")
	}
	if _, err := Quantize([]float32{1, 2, 3}, TurboQuantConfig{Bits: 4, GroupSize: 0}); err == nil {
		t.Error("GroupSize=0 must be rejected")
	}
}

func TestRotationMatrixOrthogonal(t *testing.T) {
	const dim = 32
	m := GenerateRotationMatrix(dim, 7)
	if len(m) != dim*dim {
		t.Fatalf("matrix length: got %d want %d", len(m), dim*dim)
	}

	// R times R-transpose must be the identity to within rounding.
	for i := range dim {
		for j := range dim {
			var dot float64
			for k := range dim {
				dot += float64(m[i*dim+k]) * float64(m[j*dim+k])
			}
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(dot-want) > 1e-4 {
				t.Fatalf("R Rᵀ[%d,%d] = %.6f want %.1f", i, j, dot, want)
			}
		}
	}

	// The same seed must reproduce the same matrix exactly.
	again := GenerateRotationMatrix(dim, 7)
	for i := range m {
		if m[i] != again[i] {
			t.Fatalf("seed not deterministic at %d: %v vs %v", i, m[i], again[i])
		}
	}
}
