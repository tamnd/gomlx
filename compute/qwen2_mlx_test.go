// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package compute

import (
	"math"
	"testing"

	"github.com/tamnd/gomlx/mlxgo"
)

// TestProjectBiasBroadcast checks the one numerical path Qwen2 adds over the
// other dense families: a per-output bias added to a projection. No Qwen2
// checkpoint is present on this box, so this verifies the broadcast directly,
// over both the single-stream [seq, out] shape and the batched [n, q, out]
// shape, against a hand-computed reference. With hasBias false it must reduce to
// a plain linear, leaving the other families untouched.
func TestProjectBiasBroadcast(t *testing.T) {
	const in, out = 4, 3
	// Weight stored [out, in] as a checkpoint stores it.
	w, err := mlxgo.FromFloat32([]int{out, in}, []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	})
	if err != nil {
		t.Fatalf("w: %v", err)
	}
	bias, err := mlxgo.FromFloat32([]int{out}, []float32{10, 20, 30})
	if err != nil {
		t.Fatalf("bias: %v", err)
	}

	check := func(name string, shape []int, xdata []float32) {
		x, err := mlxgo.FromFloat32(shape, xdata)
		if err != nil {
			t.Fatalf("%s x: %v", name, err)
		}
		// Reference: x @ W^T picks the first `out` columns of x (W is a partial
		// identity), then adds the bias per output channel.
		got, err := projectBias(x, w, bias, true)
		if err != nil {
			t.Fatalf("%s projectBias: %v", name, err)
		}
		vals, err := got.ToFloat32()
		if err != nil {
			t.Fatalf("%s read: %v", name, err)
		}
		rows := len(xdata) / in
		biasv := []float32{10, 20, 30}
		for r := 0; r < rows; r++ {
			for c := 0; c < out; c++ {
				want := xdata[r*in+c] + biasv[c]
				if math.Abs(float64(vals[r*out+c]-want)) > 1e-5 {
					t.Errorf("%s [%d,%d]: got %v want %v", name, r, c, vals[r*out+c], want)
				}
			}
		}

		// hasBias false must be an exact plain linear: no bias added.
		plain, err := projectBias(x, w, bias, false)
		if err != nil {
			t.Fatalf("%s plain: %v", name, err)
		}
		pv, err := plain.ToFloat32()
		if err != nil {
			t.Fatalf("%s plain read: %v", name, err)
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < out; c++ {
				if math.Abs(float64(pv[r*out+c]-xdata[r*in+c])) > 1e-5 {
					t.Errorf("%s plain [%d,%d]: got %v want %v", name, r, c, pv[r*out+c], xdata[r*in+c])
				}
			}
		}
	}

	// Single-stream: [seq, in].
	check("seq", []int{2, in}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	// Batched: [n, q, in].
	check("batch", []int{2, 1, in}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
}
