// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package mlxgo

import (
	"math"
	"testing"
)

// TestRoundTrip uploads a buffer and reads it back unchanged.
func TestRoundTrip(t *testing.T) {
	in := []float32{1, 2, 3, 4, 5, 6}
	a, err := FromFloat32([]int{2, 3}, in)
	if err != nil {
		t.Fatalf("FromFloat32: %v", err)
	}
	defer a.Free()
	out, err := a.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("length: got %d want %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Errorf("index %d: got %v want %v", i, out[i], in[i])
		}
	}
}

// TestMatMulAdd runs a real matmul and elementwise add on the GPU and checks
// the numbers against a hand computation.
func TestMatMulAdd(t *testing.T) {
	// a: 2x3, b: 3x2 -> 2x2.
	a, _ := FromFloat32([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := FromFloat32([]int{3, 2}, []float32{7, 8, 9, 10, 11, 12})
	defer a.Free()
	defer b.Free()

	prod, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	defer prod.Free()
	if got := prod.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("matmul shape: got %v", got)
	}

	// Add a bias broadcast is not used; add prod to itself to exercise Add.
	sum, err := Add(prod, prod)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	defer sum.Free()

	got, err := sum.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32: %v", err)
	}
	// a@b = [[58,64],[139,154]]; doubled = [[116,128],[278,308]].
	want := []float32{116, 128, 278, 308}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Errorf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}
