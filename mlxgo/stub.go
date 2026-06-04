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

// FromRawBytes would upload a raw little-endian buffer with the given dtype.
func FromRawBytes(shape []int, dt DType, raw []byte) (Array, error) {
	return Array{}, ErrUnavailable
}

// ToUint32 would copy a uint32 array back to host memory.
func (a Array) ToUint32() ([]uint32, error) { return nil, ErrUnavailable }

// Take would gather rows of a at the given indices (embedding lookup).
func Take(a, indices Array) (Array, error) { return Array{}, ErrUnavailable }

// Reshape would return a with a new shape.
func Reshape(a Array, shape []int) (Array, error) { return Array{}, ErrUnavailable }

// Transpose would permute the axes of a.
func Transpose(a Array, axes []int) (Array, error) { return Array{}, ErrUnavailable }

// SoftmaxAxis would softmax a along one axis.
func SoftmaxAxis(a Array, axis int) (Array, error) { return Array{}, ErrUnavailable }

// Concat would join arrays along an axis.
func Concat(arrays []Array, axis int) (Array, error) { return Array{}, ErrUnavailable }

// Multiply would compute a * b elementwise.
func Multiply(a, b Array) (Array, error) { return Array{}, ErrUnavailable }

// Silu would compute x * sigmoid(x).
func Silu(a Array) (Array, error) { return Array{}, ErrUnavailable }

// MulScalar would multiply every element of a by a host float.
func MulScalar(a Array, v float32) (Array, error) { return Array{}, ErrUnavailable }

// AddScalar would add a host float to every element of a.
func AddScalar(a Array, v float32) (Array, error) { return Array{}, ErrUnavailable }

// GeluTanh would compute the tanh approximation of GELU elementwise.
func GeluTanh(a Array) (Array, error) { return Array{}, ErrUnavailable }

// Gelu would compute the exact GELU elementwise.
func Gelu(a Array) (Array, error) { return Array{}, ErrUnavailable }

// Argmax would return the index of the maximum along an axis.
func Argmax(a Array, axis int, keepdims bool) (Array, error) { return Array{}, ErrUnavailable }

// Astype would cast a to another dtype.
func Astype(a Array, dt DType) (Array, error) { return Array{}, ErrUnavailable }

// RMSNorm would apply the fused RMS normalization.
func RMSNorm(x, weight Array, eps float32) (Array, error) { return Array{}, ErrUnavailable }

// RoPE would apply rotary position embeddings.
func RoPE(x Array, dims int, traditional bool, base, scale float32, offset int) (Array, error) {
	return Array{}, ErrUnavailable
}

// SDPA would run fused scaled dot-product attention.
func SDPA(q, k, v Array, scale float32, causal bool) (Array, error) {
	return Array{}, ErrUnavailable
}

// SDPAMasked would run fused attention with an explicit additive mask.
func SDPAMasked(q, k, v, mask Array, scale float32) (Array, error) {
	return Array{}, ErrUnavailable
}

// DType would report the element type of the array.
func (a Array) DType() DType { return F32 }

// QuantizedMatmul would multiply x by a quantized weight.
func QuantizedMatmul(x, w, scales, biases Array, transpose bool, groupSize, bits int) (Array, error) {
	return Array{}, ErrUnavailable
}

// Compile would trace a function into a single MLX graph. Without the runtime
// there is nothing to trace, so it reports the backend is unavailable.
func Compile(fn CompiledFunc, shapeless bool) (CompiledFunc, error) {
	return nil, ErrUnavailable
}
