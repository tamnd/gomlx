// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package mlxgo

import (
	"math"
	"testing"
)

// eager runs the reference computation one op at a time so the compiled graph
// has something to match: out = silu(x @ w), plus a bias add so the graph has
// more than one node.
func eager(x, w, b Array) (Array, error) {
	xw, err := MatMul(x, w)
	if err != nil {
		return Array{}, err
	}
	biased, err := Add(xw, b)
	if err != nil {
		return Array{}, err
	}
	return Silu(biased)
}

func approxEqual(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range got {
		if d := float32(math.Abs(float64(got[i] - want[i]))); d > tol {
			t.Fatalf("element %d differs: got %v want %v (|d|=%v > %v)", i, got[i], want[i], d, tol)
		}
	}
}

func TestCompileMatchesEager(t *testing.T) {
	// A small dense layer: x is 2x3, w is 3x4, bias is 1x4 (broadcast over rows).
	x, err := FromFloat32([]int{2, 3}, []float32{
		0.1, -0.2, 0.3,
		0.4, 0.5, -0.6,
	})
	if err != nil {
		t.Fatalf("x: %v", err)
	}
	w, err := FromFloat32([]int{3, 4}, []float32{
		0.10, 0.20, -0.30, 0.40,
		-0.50, 0.60, 0.70, -0.80,
		0.90, -1.00, 0.11, 0.12,
	})
	if err != nil {
		t.Fatalf("w: %v", err)
	}
	b, err := FromFloat32([]int{1, 4}, []float32{0.01, -0.02, 0.03, -0.04})
	if err != nil {
		t.Fatalf("b: %v", err)
	}

	want, err := eager(x, w, b)
	if err != nil {
		t.Fatalf("eager: %v", err)
	}
	if err := Eval(want); err != nil {
		t.Fatalf("eval eager: %v", err)
	}
	wantHost, err := want.ToFloat32()
	if err != nil {
		t.Fatalf("eager host: %v", err)
	}

	compiled, err := Compile(func(in []Array) ([]Array, error) {
		out, err := eager(in[0], in[1], in[2])
		if err != nil {
			return nil, err
		}
		return []Array{out}, nil
	}, false)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	outs, err := compiled([]Array{x, w, b})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if len(outs) != 1 {
		t.Fatalf("output count: got %d want 1", len(outs))
	}
	if err := Eval(outs[0]); err != nil {
		t.Fatalf("eval compiled: %v", err)
	}
	gotHost, err := outs[0].ToFloat32()
	if err != nil {
		t.Fatalf("compiled host: %v", err)
	}
	approxEqual(t, gotHost, wantHost, 1e-5)

	// Call again with fresh inputs to confirm the traced graph is reusable and
	// not bound to the first set of arrays.
	x2, err := FromFloat32([]int{2, 3}, []float32{
		-0.7, 0.8, -0.9,
		1.0, -1.1, 1.2,
	})
	if err != nil {
		t.Fatalf("x2: %v", err)
	}
	want2, err := eager(x2, w, b)
	if err != nil {
		t.Fatalf("eager2: %v", err)
	}
	if err := Eval(want2); err != nil {
		t.Fatalf("eval eager2: %v", err)
	}
	want2Host, err := want2.ToFloat32()
	if err != nil {
		t.Fatalf("eager2 host: %v", err)
	}

	outs2, err := compiled([]Array{x2, w, b})
	if err != nil {
		t.Fatalf("apply2: %v", err)
	}
	if err := Eval(outs2[0]); err != nil {
		t.Fatalf("eval compiled2: %v", err)
	}
	got2Host, err := outs2[0].ToFloat32()
	if err != nil {
		t.Fatalf("compiled2 host: %v", err)
	}
	approxEqual(t, got2Host, want2Host, 1e-5)
}
