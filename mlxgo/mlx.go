// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package mlxgo

/*
#cgo CFLAGS: -I${SRCDIR}/../third_party/mlx-c/include
#cgo LDFLAGS: -L${SRCDIR}/../third_party/mlx-c/lib -lmlxc -lmlx -lc++

#include <stdlib.h>
#include "mlx/c/mlx.h"

// new_data uploads a row-major float32 buffer as an mlx_array. The shape is
// passed as C ints; mlx copies the data, so the Go buffer can be reused after.
static mlx_array gomlx_from_f32(const float* data, const int* shape, int ndim) {
    return mlx_array_new_data((const void*)data, shape, ndim, MLX_FLOAT32);
}
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// Available reports that the MLX runtime is linked into this binary.
const Available = true

// Stream wraps an mlx_stream. The GPU stream is the device queue every op runs
// on; we hold one default GPU stream for the process.
type Stream struct {
	s C.mlx_stream
}

// NewStream returns the default GPU stream.
func NewStream() (Stream, error) {
	return Stream{s: C.mlx_default_gpu_stream()}, nil
}

// arrayHandle recovers the C handle stored in an Array.
func arrayHandle(a Array) C.mlx_array {
	return *(*C.mlx_array)(a.ptr)
}

// wrap boxes a C handle and shape into an Array, attaching a finalizer that
// frees the device memory if the caller forgets to.
func wrap(h C.mlx_array, shape []int) Array {
	box := (*C.mlx_array)(C.malloc(C.size_t(unsafe.Sizeof(h))))
	*box = h
	a := Array{ptr: unsafe.Pointer(box), shape: shape}
	runtime.SetFinalizer(box, func(p *C.mlx_array) {
		C.mlx_array_free(*p)
		C.free(unsafe.Pointer(p))
	})
	return a
}

// FromFloat32 uploads a row-major float32 buffer to the device.
func FromFloat32(shape []int, data []float32) (Array, error) {
	cshape := make([]C.int, len(shape))
	for i, d := range shape {
		cshape[i] = C.int(d)
	}
	var dptr *C.float
	if len(data) > 0 {
		dptr = (*C.float)(unsafe.Pointer(&data[0]))
	}
	var sptr *C.int
	if len(cshape) > 0 {
		sptr = &cshape[0]
	}
	h := C.gomlx_from_f32(dptr, sptr, C.int(len(shape)))
	return wrap(h, shape), nil
}

// ToFloat32 evaluates the array and copies it back to host memory.
func (a Array) ToFloat32() ([]float32, error) {
	h := arrayHandle(a)
	if err := evalHandles(h); err != nil {
		return nil, err
	}
	n := a.NumElements()
	out := make([]float32, n)
	src := C.mlx_array_data_float32(h)
	if n > 0 && src != nil {
		C.memcpy(unsafe.Pointer(&out[0]), unsafe.Pointer(src), C.size_t(n*4))
	}
	return out, nil
}

// MatMul computes a @ b on the default stream.
func MatMul(a, b Array) (Array, error) {
	var res C.mlx_array
	C.mlx_matmul(&res, arrayHandle(a), arrayHandle(b), C.mlx_default_gpu_stream())
	return wrap(res, matmulShape(a.shape, b.shape)), nil
}

// Add computes a + b on the default stream.
func Add(a, b Array) (Array, error) {
	var res C.mlx_array
	C.mlx_add(&res, arrayHandle(a), arrayHandle(b), C.mlx_default_gpu_stream())
	return wrap(res, a.shape), nil
}

// Eval forces evaluation of the lazy graph for the given arrays.
func Eval(arrays ...Array) error {
	hs := make([]C.mlx_array, len(arrays))
	for i, a := range arrays {
		hs[i] = arrayHandle(a)
	}
	return evalHandles(hs...)
}

func evalHandles(hs ...C.mlx_array) error {
	for _, h := range hs {
		C.mlx_eval(h)
	}
	return nil
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
	C.free(a.ptr)
}

// matmulShape derives the result shape of a 2D-or-batched matmul, leaving the
// leading batch dimensions of a in place and taking the last dim from b.
func matmulShape(as, bs []int) []int {
	if len(as) == 0 || len(bs) == 0 {
		return as
	}
	out := append([]int(nil), as...)
	out[len(out)-1] = bs[len(bs)-1]
	return out
}
