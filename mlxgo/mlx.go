// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package mlxgo

/*
#cgo CFLAGS: -I${SRCDIR}/../third_party/mlx-c/include
#cgo LDFLAGS: -L${SRCDIR}/../third_party/mlx-c/lib -lmlxc -lmlx -framework Metal -framework Foundation -framework Accelerate -lc++

#include <stdlib.h>
#include <string.h>
#include "mlx/c/mlx.h"

// from_f32 uploads a row-major float32 buffer as an mlx_array. mlx copies the
// data, so the Go buffer can be reused after the call returns.
static mlx_array gomlx_from_f32(const float* data, const int* shape, int ndim) {
    return mlx_array_new_data((const void*)data, shape, ndim, MLX_FLOAT32);
}

// eval_one forces evaluation of a single array by wrapping it in a vector.
static int gomlx_eval_one(mlx_array a) {
    mlx_vector_array v = mlx_vector_array_new_value(a);
    int rc = mlx_eval(v);
    mlx_vector_array_free(v);
    return rc;
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

// Stream wraps an mlx_stream. The GPU stream is the device queue every op runs
// on; we hold one default GPU stream for the process.
type Stream struct {
	s C.mlx_stream
}

// gpuStream is the process-wide default GPU stream, fetched once. mlx tracks a
// single default GPU stream, so every op shares it.
var gpuStream = C.mlx_default_gpu_stream_new()

// NewStream returns the default GPU stream.
func NewStream() (Stream, error) {
	return Stream{s: gpuStream}, nil
}

// arrayHandle recovers the C handle stored in an Array.
func arrayHandle(a Array) C.mlx_array {
	return *(*C.mlx_array)(a.ptr)
}

// wrap boxes a C handle and shape into an Array, attaching a finalizer that
// frees the device memory if the caller forgets to. The box is a Go allocation
// (SetFinalizer requires that); it only holds a C pointer, which cgo permits.
func wrap(h C.mlx_array, shape []int) Array {
	box := new(C.mlx_array)
	*box = h
	a := Array{ptr: unsafe.Pointer(box), shape: shape}
	runtime.SetFinalizer(box, func(p *C.mlx_array) {
		C.mlx_array_free(*p)
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
	if rc := C.gomlx_eval_one(h); rc != 0 {
		return nil, fmt.Errorf("mlxgo: eval failed (rc=%d)", int(rc))
	}
	n := a.NumElements()
	out := make([]float32, n)
	src := C.mlx_array_data_float32(h)
	if n > 0 && src != nil {
		C.memcpy(unsafe.Pointer(&out[0]), unsafe.Pointer(src), C.size_t(n*4))
	}
	return out, nil
}

// MatMul computes a @ b on the default GPU stream.
func MatMul(a, b Array) (Array, error) {
	var res C.mlx_array
	if rc := C.mlx_matmul(&res, arrayHandle(a), arrayHandle(b), gpuStream); rc != 0 {
		return Array{}, fmt.Errorf("mlxgo: matmul failed (rc=%d)", int(rc))
	}
	return wrap(res, matmulShape(a.shape, b.shape)), nil
}

// Add computes a + b on the default GPU stream.
func Add(a, b Array) (Array, error) {
	var res C.mlx_array
	if rc := C.mlx_add(&res, arrayHandle(a), arrayHandle(b), gpuStream); rc != 0 {
		return Array{}, fmt.Errorf("mlxgo: add failed (rc=%d)", int(rc))
	}
	return wrap(res, a.shape), nil
}

// Eval forces evaluation of the lazy graph for the given arrays.
func Eval(arrays ...Array) error {
	for _, a := range arrays {
		if rc := C.gomlx_eval_one(arrayHandle(a)); rc != 0 {
			return fmt.Errorf("mlxgo: eval failed (rc=%d)", int(rc))
		}
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
