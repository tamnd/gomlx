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

func approx(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Errorf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func mustF32(t *testing.T, shape []int, data []float32) Array {
	t.Helper()
	a, err := FromFloat32(shape, data)
	if err != nil {
		t.Fatalf("FromFloat32: %v", err)
	}
	return a
}

func TestTakeEmbedding(t *testing.T) {
	table := mustF32(t, []int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	idx, err := FromRawBytes([]int{2}, I32, []byte{2, 0, 0, 0, 0, 0, 0, 0})
	if err != nil {
		t.Fatalf("indices: %v", err)
	}
	out, err := Take(table, idx)
	if err != nil {
		t.Fatalf("Take: %v", err)
	}
	got, _ := out.ToFloat32()
	approx(t, got, []float32{5, 6, 1, 2}, 1e-6)
	if s := out.Shape(); len(s) != 2 || s[0] != 2 || s[1] != 2 {
		t.Errorf("shape: got %v", s)
	}
}

func TestSilu(t *testing.T) {
	x := mustF32(t, []int{3}, []float32{0, 1, -1})
	out, err := Silu(x)
	if err != nil {
		t.Fatalf("Silu: %v", err)
	}
	got, _ := out.ToFloat32()
	approx(t, got, []float32{0, 0.7310586, -0.26894143}, 1e-5)
}

// geluTanhRef is the host reference for the tanh GELU approximation, used to
// check the device op.
func geluTanhRef(x float64) float64 {
	inner := math.Sqrt(2/math.Pi) * (x + 0.044715*x*x*x)
	return 0.5 * x * (1 + math.Tanh(inner))
}

func TestGeluTanh(t *testing.T) {
	in := []float32{0, 1, -1, 2, -2, 0.5}
	x := mustF32(t, []int{len(in)}, in)
	out, err := GeluTanh(x)
	if err != nil {
		t.Fatalf("GeluTanh: %v", err)
	}
	got, _ := out.ToFloat32()
	want := make([]float32, len(in))
	for i, v := range in {
		want[i] = float32(geluTanhRef(float64(v)))
	}
	approx(t, got, want, 1e-5)
}

func TestMulScalar(t *testing.T) {
	x := mustF32(t, []int{4}, []float32{1, 2, 3, 4})
	out, err := MulScalar(x, 2.5)
	if err != nil {
		t.Fatalf("MulScalar: %v", err)
	}
	got, _ := out.ToFloat32()
	approx(t, got, []float32{2.5, 5, 7.5, 10}, 1e-5)
}

func TestAddScalar(t *testing.T) {
	x := mustF32(t, []int{3}, []float32{-1, 0, 1})
	out, err := AddScalar(x, 1.0)
	if err != nil {
		t.Fatalf("AddScalar: %v", err)
	}
	got, _ := out.ToFloat32()
	approx(t, got, []float32{0, 1, 2}, 1e-6)
}

func TestSoftmaxAxis(t *testing.T) {
	x := mustF32(t, []int{2, 2}, []float32{1, 1, 0, 2})
	out, err := SoftmaxAxis(x, 1)
	if err != nil {
		t.Fatalf("SoftmaxAxis: %v", err)
	}
	got, _ := out.ToFloat32()
	// Row 0: equal -> 0.5,0.5. Row 1: softmax(0,2).
	e0, e2 := 1.0, math.Exp(2)
	approx(t, got, []float32{0.5, 0.5, float32(e0 / (e0 + e2)), float32(e2 / (e0 + e2))}, 1e-5)
}

func TestArgmaxAxis(t *testing.T) {
	x := mustF32(t, []int{2, 3}, []float32{1, 5, 2, 7, 0, 3})
	out, err := Argmax(x, 1, false)
	if err != nil {
		t.Fatalf("Argmax: %v", err)
	}
	got, _ := out.ToUint32()
	if len(got) != 2 || got[0] != 1 || got[1] != 0 {
		t.Errorf("argmax: got %v want [1 0]", got)
	}
}

func TestReshapeTranspose(t *testing.T) {
	x := mustF32(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	r, err := Reshape(x, []int{3, 2})
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	if s := r.Shape(); s[0] != 3 || s[1] != 2 {
		t.Errorf("reshape shape: %v", s)
	}
	tr, err := Transpose(x, []int{1, 0})
	if err != nil {
		t.Fatalf("Transpose: %v", err)
	}
	got, _ := tr.ToFloat32()
	// Transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]].
	approx(t, got, []float32{1, 4, 2, 5, 3, 6}, 1e-6)
}

func TestConcat(t *testing.T) {
	a := mustF32(t, []int{1, 2}, []float32{1, 2})
	b := mustF32(t, []int{1, 2}, []float32{3, 4})
	out, err := Concat([]Array{a, b}, 0)
	if err != nil {
		t.Fatalf("Concat: %v", err)
	}
	got, _ := out.ToFloat32()
	approx(t, got, []float32{1, 2, 3, 4}, 1e-6)
	if s := out.Shape(); s[0] != 2 || s[1] != 2 {
		t.Errorf("concat shape: %v", s)
	}
}

func TestRMSNorm(t *testing.T) {
	x := mustF32(t, []int{1, 2}, []float32{3, 4})
	w := mustF32(t, []int{2}, []float32{1, 1})
	out, err := RMSNorm(x, w, 1e-6)
	if err != nil {
		t.Fatalf("RMSNorm: %v", err)
	}
	got, _ := out.ToFloat32()
	// rms = sqrt(mean(9,16)) = sqrt(12.5) = 3.535534.
	rms := math.Sqrt(12.5)
	approx(t, got, []float32{float32(3 / rms), float32(4 / rms)}, 1e-4)
}

func TestRoPEIdentityAtZero(t *testing.T) {
	// A single position at offset 0 rotates by angle 0, so the output equals
	// the input. Shape is [batch, heads, seq, head_dim] = [1,1,1,4].
	in := []float32{0.1, 0.2, 0.3, 0.4}
	x := mustF32(t, []int{1, 1, 1, 4}, in)
	out, err := RoPE(x, 4, false, 1000000, 1, 0)
	if err != nil {
		t.Fatalf("RoPE: %v", err)
	}
	got, _ := out.ToFloat32()
	approx(t, got, in, 1e-5)
}

func TestSDPAShapeAndRun(t *testing.T) {
	// One batch, one head, two keys, head_dim 4.
	q := mustF32(t, []int{1, 1, 1, 4}, []float32{1, 0, 0, 0})
	k := mustF32(t, []int{1, 1, 2, 4}, []float32{1, 0, 0, 0, 0, 1, 0, 0})
	v := mustF32(t, []int{1, 1, 2, 4}, []float32{10, 20, 30, 40, 50, 60, 70, 80})
	out, err := SDPA(q, k, v, 1.0, false)
	if err != nil {
		t.Fatalf("SDPA: %v", err)
	}
	got, _ := out.ToFloat32()
	if s := out.Shape(); len(s) != 4 || s[3] != 4 {
		t.Errorf("sdpa shape: got %v", s)
	}
	// Scores before softmax: q.k = [1, 0]; softmax([1,0]) = [w0, w1].
	w0 := math.Exp(1) / (math.Exp(1) + 1)
	w1 := 1 / (math.Exp(1) + 1)
	want := []float32{
		float32(w0*10 + w1*50), float32(w0*20 + w1*60),
		float32(w0*30 + w1*70), float32(w0*40 + w1*80),
	}
	approx(t, got, want, 1e-4)
}
