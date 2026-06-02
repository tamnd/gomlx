// SPDX-License-Identifier: Apache-2.0

//go:build !mlx

package mlxgo

// Available reports whether the MLX runtime is linked into this binary. It is
// false in the stub build (no "mlx" tag), which lets the engine refuse to start
// the GPU path with a clear message instead of crashing.
const Available = false

// Stream is a no-op handle in the stub build.
type Stream struct{}

// NewStream returns an error: there is no device without MLX.
func NewStream() (Stream, error) { return Stream{}, ErrUnavailable }

// FromFloat32 would upload a row-major float32 buffer to the device.
func FromFloat32(shape []int, data []float32) (Array, error) {
	return Array{}, ErrUnavailable
}

// ToFloat32 would copy an array back to host memory.
func (a Array) ToFloat32() ([]float32, error) { return nil, ErrUnavailable }

// MatMul would compute a @ b on the device.
func MatMul(a, b Array) (Array, error) { return Array{}, ErrUnavailable }

// Add would compute a + b on the device.
func Add(a, b Array) (Array, error) { return Array{}, ErrUnavailable }

// Eval would force evaluation of the lazy graph for the given arrays.
func Eval(arrays ...Array) error { return ErrUnavailable }

// Free would release the device memory backing an array.
func (a Array) Free() {}
