// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package mlxgo

/*
#cgo CFLAGS: -I${SRCDIR}/../third_party/mlx-c/include
#cgo LDFLAGS: -L${SRCDIR}/../third_party/mlx-c/lib -lmlxc -lmlx -framework Metal -framework Foundation -framework Accelerate -lc++

#include <stdlib.h>
#include <string.h>
#include "mlx/c/mlx.h"

// from_data uploads a raw buffer of the given dtype as an mlx_array. mlx copies
// the data, so the Go buffer can be reused after the call returns.
static mlx_array gomlx_from_data(const void* data, const int* shape, int ndim, mlx_dtype dt) {
    return mlx_array_new_data(data, shape, ndim, dt);
}

// eval_one forces evaluation of a single array by wrapping it in a vector.
static int gomlx_eval_one(mlx_array a) {
    mlx_vector_array v = mlx_vector_array_new_value(a);
    int rc = mlx_eval(v);
    mlx_vector_array_free(v);
    return rc;
}

// rope wraps mlx_fast_rope, building the optional base on the C side.
static int gomlx_rope(mlx_array* res, mlx_array x, int dims, bool traditional,
                      float base, float scale, int offset, mlx_stream s) {
    mlx_optional_float b = {base, true};
    mlx_array no_freqs = mlx_array_new();
    int rc = mlx_fast_rope(res, x, dims, traditional, b, scale, offset, no_freqs, s);
    mlx_array_free(no_freqs);
    return rc;
}

// sdpa wraps the fused attention, passing a mask mode string and no mask array.
static int gomlx_sdpa(mlx_array* res, mlx_array q, mlx_array k, mlx_array v,
                      float scale, const char* mode, mlx_stream s) {
    mlx_vector_array no_mask = mlx_vector_array_new();
    int rc = mlx_fast_scaled_dot_product_attention(res, q, k, v, scale, mode, no_mask, s);
    mlx_vector_array_free(no_mask);
    return rc;
}

// sdpa_mask wraps the fused attention with an explicit additive mask array. The
// mask is added to the attention scores before softmax, so padding positions
// carry a large negative value and contribute nothing.
static int gomlx_sdpa_mask(mlx_array* res, mlx_array q, mlx_array k, mlx_array v,
                           float scale, mlx_array mask, mlx_stream s) {
    mlx_vector_array mv = mlx_vector_array_new_value(mask);
    int rc = mlx_fast_scaled_dot_product_attention(res, q, k, v, scale, "array", mv, s);
    mlx_vector_array_free(mv);
    return rc;
}

// mul_scalar multiplies an array by a host float, broadcasting the scalar over
// every element. It builds the scalar array on the C side so the Go wrapper
// stays a single call and the temporary is freed here.
static int gomlx_mul_scalar(mlx_array* res, mlx_array a, float v, mlx_stream s) {
    mlx_array c = mlx_array_new_float32(v);
    int rc = mlx_multiply(res, a, c, s);
    mlx_array_free(c);
    return rc;
}

// add_scalar adds a host float to every element of an array.
static int gomlx_add_scalar(mlx_array* res, mlx_array a, float v, mlx_stream s) {
    mlx_array c = mlx_array_new_float32(v);
    int rc = mlx_add(res, a, c, s);
    mlx_array_free(c);
    return rc;
}

// gelu_tanh computes the tanh approximation of GELU,
//   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )),
// which is the "gelu_pytorch_tanh" activation some dense families use in their
// MLP. mlx-c has no fused gelu, so it is built from elementwise primitives; the
// scalar constants are created once and every intermediate is freed before the
// call returns.
static int gomlx_gelu_tanh(mlx_array* res, mlx_array x, mlx_stream s) {
    mlx_array kappa = mlx_array_new_float32(0.044715f);
    mlx_array beta  = mlx_array_new_float32(0.7978845608028654f); // sqrt(2/pi)
    mlx_array one   = mlx_array_new_float32(1.0f);
    mlx_array half  = mlx_array_new_float32(0.5f);
    mlx_array x2 = mlx_array_new();
    mlx_array x3 = mlx_array_new();
    mlx_array kx3 = mlx_array_new();
    mlx_array inner = mlx_array_new();
    mlx_array scaled = mlx_array_new();
    mlx_array t = mlx_array_new();
    mlx_array onePlus = mlx_array_new();
    mlx_array halfx = mlx_array_new();
    int rc = 0;
    if ((rc = mlx_multiply(&x2, x, x, s))) goto done;
    if ((rc = mlx_multiply(&x3, x2, x, s))) goto done;
    if ((rc = mlx_multiply(&kx3, kappa, x3, s))) goto done;
    if ((rc = mlx_add(&inner, x, kx3, s))) goto done;
    if ((rc = mlx_multiply(&scaled, beta, inner, s))) goto done;
    if ((rc = mlx_tanh(&t, scaled, s))) goto done;
    if ((rc = mlx_add(&onePlus, one, t, s))) goto done;
    if ((rc = mlx_multiply(&halfx, half, x, s))) goto done;
    rc = mlx_multiply(res, halfx, onePlus, s);
done:
    mlx_array_free(kappa); mlx_array_free(beta); mlx_array_free(one);
    mlx_array_free(half); mlx_array_free(x2); mlx_array_free(x3);
    mlx_array_free(kx3); mlx_array_free(inner); mlx_array_free(scaled);
    mlx_array_free(t); mlx_array_free(onePlus); mlx_array_free(halfx);
    return rc;
}

// gomlxCompileTrampoline is the Go callback that traces a compiled function. It
// is declared here so the closure builder below can take its address; the
// definition lives in compile.go (an //export file cannot also define C
// functions in its preamble). gomlx_new_compile_closure has external linkage so
// compile.go can call it through an extern declaration.
extern int gomlxCompileTrampoline(mlx_vector_array* res, const mlx_vector_array input, void* payload);
mlx_closure gomlx_new_compile_closure(void* payload) {
    return mlx_closure_new_func_payload(gomlxCompileTrampoline, payload, free);
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Available reports that the MLX runtime is linked into this binary.
const Available = true

// Stream wraps an mlx_stream.
type Stream struct {
	s C.mlx_stream
}

// gpuStream is the process-wide default GPU stream, fetched once.
var gpuStream = C.mlx_default_gpu_stream_new()

// NewStream returns the default GPU stream.
func NewStream() (Stream, error) { return Stream{s: gpuStream}, nil }

// dtypeC maps a binding DType to the MLX dtype enum.
func dtypeC(dt DType) C.mlx_dtype {
	switch dt {
	case F16:
		return C.MLX_FLOAT16
	case BF16:
		return C.MLX_BFLOAT16
	case U32:
		return C.MLX_UINT32
	case I32:
		return C.MLX_INT32
	default:
		return C.MLX_FLOAT32
	}
}

// arrayHandle recovers the C handle stored in an Array.
func arrayHandle(a Array) C.mlx_array { return *(*C.mlx_array)(a.ptr) }

// DType reports the element type of the array.
func (a Array) DType() DType {
	switch C.mlx_array_dtype(arrayHandle(a)) {
	case C.MLX_FLOAT16:
		return F16
	case C.MLX_BFLOAT16:
		return BF16
	case C.MLX_UINT32:
		return U32
	case C.MLX_INT32:
		return I32
	default:
		return F32
	}
}

// handleShape reads the true shape of an mlx array. mlx computes shapes eagerly
// even when values are lazy, so this is valid before evaluation.
func handleShape(h C.mlx_array) []int {
	nd := int(C.mlx_array_ndim(h))
	if nd == 0 {
		return nil
	}
	sp := C.mlx_array_shape(h)
	out := make([]int, nd)
	for i := 0; i < nd; i++ {
		out[i] = int(*(*C.int)(unsafe.Add(unsafe.Pointer(sp), i*int(unsafe.Sizeof(C.int(0))))))
	}
	return out
}

// wrap boxes a C handle into an Array and reads its shape from MLX, so no op
// needs to track shapes by hand. The box is a Go allocation (SetFinalizer
// requires that) holding only a C pointer, which cgo permits.
func wrap(h C.mlx_array) Array {
	box := new(C.mlx_array)
	*box = h
	a := Array{ptr: unsafe.Pointer(box), shape: handleShape(h)}
	runtime.SetFinalizer(box, func(p *C.mlx_array) { C.mlx_array_free(*p) })
	return a
}

// check turns a non-zero mlx return code into an error.
func check(rc C.int, op string) error {
	if rc != 0 {
		return fmt.Errorf("mlxgo: %s failed (rc=%d)", op, int(rc))
	}
	return nil
}

func cints(xs []int) (*C.int, []C.int) {
	if len(xs) == 0 {
		return nil, nil
	}
	c := make([]C.int, len(xs))
	for i, x := range xs {
		c[i] = C.int(x)
	}
	return &c[0], c
}

// FromFloat32 uploads a row-major float32 buffer to the device.
func FromFloat32(shape []int, data []float32) (Array, error) {
	var dptr unsafe.Pointer
	if len(data) > 0 {
		dptr = unsafe.Pointer(&data[0])
	}
	sptr, _ := cints(shape)
	h := C.gomlx_from_data(dptr, sptr, C.int(len(shape)), C.MLX_FLOAT32)
	return wrap(h), nil
}

// FromRawBytes uploads a raw little-endian buffer with the given dtype. It is
// the path weight loading uses: safetensors hands over bytes plus a dtype.
func FromRawBytes(shape []int, dt DType, raw []byte) (Array, error) {
	var dptr unsafe.Pointer
	if len(raw) > 0 {
		dptr = unsafe.Pointer(&raw[0])
	}
	sptr, _ := cints(shape)
	h := C.gomlx_from_data(dptr, sptr, C.int(len(shape)), dtypeC(dt))
	return wrap(h), nil
}

// contiguousEval makes a contiguous copy of a and evaluates it, so the host
// read sees logical (row-major) order even when a is a strided view such as a
// transpose. The caller must free the returned handle.
func contiguousEval(a Array) (C.mlx_array, error) {
	var cont C.mlx_array
	if err := check(C.mlx_contiguous(&cont, arrayHandle(a), false, gpuStream), "contiguous"); err != nil {
		return cont, err
	}
	if err := check(C.gomlx_eval_one(cont), "eval"); err != nil {
		C.mlx_array_free(cont)
		return cont, err
	}
	return cont, nil
}

// ToFloat32 evaluates the array and copies it back to host memory in logical
// order.
func (a Array) ToFloat32() ([]float32, error) {
	cont, err := contiguousEval(a)
	if err != nil {
		return nil, err
	}
	defer C.mlx_array_free(cont)
	n := a.NumElements()
	out := make([]float32, n)
	if src := C.mlx_array_data_float32(cont); n > 0 && src != nil {
		C.memcpy(unsafe.Pointer(&out[0]), unsafe.Pointer(src), C.size_t(n*4))
	}
	return out, nil
}

// ToUint32 evaluates the array and copies it back as uint32 (argmax results,
// token ids).
func (a Array) ToUint32() ([]uint32, error) {
	cont, err := contiguousEval(a)
	if err != nil {
		return nil, err
	}
	defer C.mlx_array_free(cont)
	n := a.NumElements()
	out := make([]uint32, n)
	if src := C.mlx_array_data_uint32(cont); n > 0 && src != nil {
		C.memcpy(unsafe.Pointer(&out[0]), unsafe.Pointer(src), C.size_t(n*4))
	}
	return out, nil
}

// Eval forces evaluation of the lazy graph for the given arrays.
func Eval(arrays ...Array) error {
	for _, a := range arrays {
		if err := check(C.gomlx_eval_one(arrayHandle(a)), "eval"); err != nil {
			return err
		}
	}
	return nil
}

// MatMul computes a @ b on the default GPU stream.
func MatMul(a, b Array) (Array, error) {
	var res C.mlx_array
	if err := check(C.mlx_matmul(&res, arrayHandle(a), arrayHandle(b), gpuStream), "matmul"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Add computes a + b on the default GPU stream.
func Add(a, b Array) (Array, error) {
	var res C.mlx_array
	if err := check(C.mlx_add(&res, arrayHandle(a), arrayHandle(b), gpuStream), "add"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Multiply computes a * b elementwise.
func Multiply(a, b Array) (Array, error) {
	var res C.mlx_array
	if err := check(C.mlx_multiply(&res, arrayHandle(a), arrayHandle(b), gpuStream), "multiply"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Silu computes x * sigmoid(x), the SwiGLU activation.
func Silu(a Array) (Array, error) {
	var sig C.mlx_array
	if err := check(C.mlx_sigmoid(&sig, arrayHandle(a), gpuStream), "sigmoid"); err != nil {
		return Array{}, err
	}
	defer C.mlx_array_free(sig)
	var res C.mlx_array
	if err := check(C.mlx_multiply(&res, arrayHandle(a), sig, gpuStream), "silu mul"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// MulScalar multiplies every element of a by the host float v. It is the
// embedding scale step for families that multiply the token embeddings by
// sqrt(hidden_size) before the first layer.
func MulScalar(a Array, v float32) (Array, error) {
	var res C.mlx_array
	if err := check(C.gomlx_mul_scalar(&res, arrayHandle(a), C.float(v), gpuStream), "mul_scalar"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// AddScalar adds the host float v to every element of a. It is used to fold the
// (1 + weight) RMSNorm convention into the loaded norm weights once at load
// time, so the fused RMSNorm can keep multiplying by a plain weight.
func AddScalar(a Array, v float32) (Array, error) {
	var res C.mlx_array
	if err := check(C.gomlx_add_scalar(&res, arrayHandle(a), C.float(v), gpuStream), "add_scalar"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// GeluTanh computes the tanh approximation of GELU elementwise, the
// gelu_pytorch_tanh activation used in some dense MLPs.
func GeluTanh(a Array) (Array, error) {
	var res C.mlx_array
	if err := check(C.gomlx_gelu_tanh(&res, arrayHandle(a), gpuStream), "gelu_tanh"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Take gathers rows of a at the given integer indices along axis 0 (embedding
// lookup). It uses take-along-axis rather than the flattened take so a row of
// width D maps to a row, not a single element.
func Take(a, indices Array) (Array, error) {
	var res C.mlx_array
	if err := check(C.mlx_take_axis(&res, arrayHandle(a), arrayHandle(indices), 0, gpuStream), "take"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Reshape returns a with a new shape.
func Reshape(a Array, shape []int) (Array, error) {
	sptr, c := cints(shape)
	var res C.mlx_array
	if err := check(C.mlx_reshape(&res, arrayHandle(a), sptr, C.size_t(len(c)), gpuStream), "reshape"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Transpose permutes the axes of a.
func Transpose(a Array, axes []int) (Array, error) {
	sptr, c := cints(axes)
	var res C.mlx_array
	if err := check(C.mlx_transpose_axes(&res, arrayHandle(a), sptr, C.size_t(len(c)), gpuStream), "transpose"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// SoftmaxAxis softmaxes a along one axis.
func SoftmaxAxis(a Array, axis int) (Array, error) {
	var res C.mlx_array
	if err := check(C.mlx_softmax_axis(&res, arrayHandle(a), C.int(axis), true, gpuStream), "softmax"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Concat joins arrays along an axis.
func Concat(arrays []Array, axis int) (Array, error) {
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, a := range arrays {
		C.mlx_vector_array_append_value(vec, arrayHandle(a))
	}
	var res C.mlx_array
	if err := check(C.mlx_concatenate_axis(&res, vec, C.int(axis), gpuStream), "concat"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Argmax returns the index of the maximum along an axis.
func Argmax(a Array, axis int, keepdims bool) (Array, error) {
	var res C.mlx_array
	if err := check(C.mlx_argmax_axis(&res, arrayHandle(a), C.int(axis), C.bool(keepdims), gpuStream), "argmax"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Astype casts a to another dtype.
func Astype(a Array, dt DType) (Array, error) {
	var res C.mlx_array
	if err := check(C.mlx_astype(&res, arrayHandle(a), dtypeC(dt), gpuStream), "astype"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// RMSNorm applies the fused RMS normalization with the given weight and epsilon.
func RMSNorm(x, weight Array, eps float32) (Array, error) {
	var res C.mlx_array
	if err := check(C.mlx_fast_rms_norm(&res, arrayHandle(x), arrayHandle(weight), C.float(eps), gpuStream), "rms_norm"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// RoPE applies rotary position embeddings, rotating dims dimensions with the
// given base and starting position offset.
func RoPE(x Array, dims int, traditional bool, base, scale float32, offset int) (Array, error) {
	var res C.mlx_array
	rc := C.gomlx_rope(&res, arrayHandle(x), C.int(dims), C.bool(traditional),
		C.float(base), C.float(scale), C.int(offset), gpuStream)
	if err := check(rc, "rope"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// SDPA runs fused scaled dot-product attention. When causal is true a causal
// mask is applied; otherwise attention is unmasked.
func SDPA(q, k, v Array, scale float32, causal bool) (Array, error) {
	mode := C.CString("")
	if causal {
		C.free(unsafe.Pointer(mode))
		mode = C.CString("causal")
	}
	defer C.free(unsafe.Pointer(mode))
	var res C.mlx_array
	rc := C.gomlx_sdpa(&res, arrayHandle(q), arrayHandle(k), arrayHandle(v), C.float(scale), mode, gpuStream)
	if err := check(rc, "sdpa"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// SDPAMasked runs fused scaled dot-product attention with an explicit additive
// mask. The mask must broadcast to [batch, heads, q_len, k_len]; valid
// positions are 0 and masked positions a large negative value. This is the
// batched-decode path, where sequences in the batch share a padded KV cache and
// the mask hides each sequence's padding.
func SDPAMasked(q, k, v, mask Array, scale float32) (Array, error) {
	var res C.mlx_array
	rc := C.gomlx_sdpa_mask(&res, arrayHandle(q), arrayHandle(k), arrayHandle(v),
		C.float(scale), arrayHandle(mask), gpuStream)
	if err := check(rc, "sdpa_mask"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// QuantizedMatmul multiplies x by a quantized weight described by scales and
// biases. transpose matches how the weight is stored; groupSize and bits come
// from the model's quantization config.
func QuantizedMatmul(x, w, scales, biases Array, transpose bool, groupSize, bits int) (Array, error) {
	var res C.mlx_array
	rc := C.mlx_quantized_matmul(&res, arrayHandle(x), arrayHandle(w), arrayHandle(scales),
		arrayHandle(biases), C.bool(transpose), C.int(groupSize), C.int(bits), gpuStream)
	if err := check(rc, "quantized_matmul"); err != nil {
		return Array{}, err
	}
	return wrap(res), nil
}

// Free releases the device memory backing an array immediately and clears the
// finalizer.
func (a Array) Free() {
	if a.ptr == nil {
		return
	}
	box := (*C.mlx_array)(a.ptr)
	runtime.SetFinalizer(box, nil)
	C.mlx_array_free(*box)
}
