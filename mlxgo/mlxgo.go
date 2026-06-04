// SPDX-License-Identifier: Apache-2.0

// Package mlxgo is the thin binding layer over the MLX C API. The real binding
// links against the mlx-c shared library through cgo and is compiled only with
// the "mlx" build tag; the default build provides a stub so the rest of the
// module compiles, vets, and tests on machines and CI runners that have no MLX
// installed. Code that needs the GPU checks Available and degrades gracefully
// rather than failing to build.
//
// The type definitions live here, untagged, so both the real and stub function
// bodies share one surface. An Array holds an opaque device handle plus the
// shape we track on the Go side; the handle is nil in the stub.
package mlxgo

import (
	"errors"
	"unsafe"
)

// ErrUnavailable is returned by every operation in the stub build. It means the
// binary was compiled without the "mlx" build tag, so no MLX runtime is linked.
var ErrUnavailable = errors.New("mlxgo: built without the mlx tag; rebuild with -tags mlx and an MLX runtime")

// DType names the element types the binding can create arrays of. The values
// are a small subset of MLX dtypes: the float types used for activations and
// weights, plus the integer types used for token ids, argmax results, and
// packed quantized weights.
type DType int

const (
	F32 DType = iota
	F16
	BF16
	U32
	I32
)

// Array is a handle to an MLX array. ptr is the underlying mlx_array in the real
// build and nil in the stub. shape is tracked on the Go side so callers can plan
// without round-tripping to the device.
type Array struct {
	ptr   unsafe.Pointer
	shape []int
}

// CompiledFunc runs a traced compute graph over a fixed list of input arrays and
// returns the output arrays. Compile produces one; calling it dispatches the
// whole graph in a single crossing into MLX instead of one op at a time.
type CompiledFunc func(inputs []Array) ([]Array, error)

// Shape returns the array's dimensions.
func (a Array) Shape() []int { return a.shape }

// NumElements returns the product of the array's dimensions.
func (a Array) NumElements() int {
	n := 1
	for _, d := range a.shape {
		n *= d
	}
	return n
}
