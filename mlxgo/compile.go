// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package mlxgo

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"

// gomlx_new_compile_closure is defined in mlx.go (a file with an //export
// directive may not also define C functions in its preamble, so the definition
// lives there and we reach it through this declaration).
extern mlx_closure gomlx_new_compile_closure(void* payload);
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// The compile registry maps an integer key to the Go function MLX should trace.
// We cannot hand a Go closure straight to C, so each compiled function is parked
// here and the key travels through the C closure payload. The trampoline reads
// the key back and looks the function up. Access is serialised because tracing
// can run on an MLX worker thread.
var (
	compileMu  sync.Mutex
	compileReg = map[int]CompiledFunc{}
	compileSeq int
)

// compiledHolder owns the C resources behind one compiled function: the source
// closure we built around the trampoline, the compiled closure MLX returned, and
// the registry key. The finalizer frees both closures (which frees the malloc'd
// payload through its destructor) and drops the registry entry, so a compiled
// function that goes out of scope cleans up after itself.
type compiledHolder struct {
	source   C.mlx_closure
	compiled C.mlx_closure
	key      int
}

func (h *compiledHolder) free() {
	C.mlx_closure_free(h.compiled)
	C.mlx_closure_free(h.source)
	compileMu.Lock()
	delete(compileReg, h.key)
	compileMu.Unlock()
}

// gomlxCompileTrampoline is the C-callable entry MLX invokes while tracing a
// compiled function. It recovers the registered Go function from the payload,
// unpacks the input arrays, runs the function to build the graph, and packs the
// outputs back into the result vector.
//
// A non-zero return is fatal: MLX's closure machinery treats it as an
// unrecoverable error and aborts, so the traced function is expected to succeed.
// The recover and the error path here are a last-resort guard that records the
// cause before the abort, which beats letting a Go panic unwind across the cgo
// boundary into undefined behaviour.
//
//export gomlxCompileTrampoline
func gomlxCompileTrampoline(res *C.mlx_vector_array, input C.mlx_vector_array, payload unsafe.Pointer) (rc C.int) {
	defer func() {
		if r := recover(); r != nil {
			compileMu.Lock()
			compileLastPanic = fmt.Sprintf("%v", r)
			compileMu.Unlock()
			rc = 1
		}
	}()

	key := int(*(*C.int)(payload))
	compileMu.Lock()
	fn := compileReg[key]
	compileMu.Unlock()
	if fn == nil {
		return 1
	}

	n := int(C.mlx_vector_array_size(input))
	inputs := make([]Array, n)
	for i := 0; i < n; i++ {
		var h C.mlx_array = C.mlx_array_new()
		if C.mlx_vector_array_get(&h, input, C.size_t(i)) != 0 {
			C.mlx_array_free(h)
			return 1
		}
		inputs[i] = wrap(h)
	}

	outputs, err := fn(inputs)
	if err != nil {
		compileMu.Lock()
		compileLastErr = err
		compileMu.Unlock()
		return 1
	}

	out := C.mlx_vector_array_new()
	for _, a := range outputs {
		if C.mlx_vector_array_append_value(out, arrayHandle(a)) != 0 {
			C.mlx_vector_array_free(out)
			return 1
		}
	}
	runtime.KeepAlive(outputs)
	C.mlx_vector_array_set(res, out)
	C.mlx_vector_array_free(out)
	return 0
}

// compileLastErr and compileLastPanic carry the most recent failure out of the
// trampoline, where a Go error or panic value has nowhere else to go. They are
// read once when an apply fails so the caller sees the real cause.
var (
	compileLastErr   error
	compileLastPanic string
)

// Compile traces fn into a single MLX graph and returns a function that runs the
// whole graph in one crossing into MLX, instead of dispatching each op eagerly
// over cgo. shapeless keeps the trace valid across input shapes; pass false to
// retrace whenever an input shape changes, which lets MLX specialise the graph.
//
// The returned function must be called with the same number of inputs fn
// expects. The trace itself is lazy: MLX runs fn on the first call.
func Compile(fn CompiledFunc, shapeless bool) (CompiledFunc, error) {
	compileMu.Lock()
	compileSeq++
	key := compileSeq
	compileReg[key] = fn
	compileMu.Unlock()

	// The payload is a malloc'd copy of the key so it survives independently of
	// any Go allocation; the closure's destructor (free) reclaims it.
	payload := C.malloc(C.size_t(unsafe.Sizeof(C.int(0))))
	*(*C.int)(payload) = C.int(key)

	source := C.gomlx_new_compile_closure(payload)
	var compiled C.mlx_closure
	if err := check(C.mlx_compile(&compiled, source, C.bool(shapeless)), "compile"); err != nil {
		C.mlx_closure_free(source)
		compileMu.Lock()
		delete(compileReg, key)
		compileMu.Unlock()
		return nil, err
	}

	h := &compiledHolder{source: source, compiled: compiled, key: key}
	runtime.SetFinalizer(h, func(x *compiledHolder) { x.free() })

	return func(inputs []Array) ([]Array, error) {
		out, err := applyCompiled(h, inputs)
		runtime.KeepAlive(h)
		return out, err
	}, nil
}

// applyCompiled feeds inputs through a compiled closure and returns its outputs.
// It packs the inputs into an MLX vector, applies the closure, and wraps each
// result back into an Array.
func applyCompiled(h *compiledHolder, inputs []Array) ([]Array, error) {
	in := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(in)
	for _, a := range inputs {
		if C.mlx_vector_array_append_value(in, arrayHandle(a)) != 0 {
			return nil, fmt.Errorf("mlxgo: compile apply: append input failed")
		}
	}
	runtime.KeepAlive(inputs)

	var out C.mlx_vector_array = C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(out)
	if rc := C.mlx_closure_apply(&out, h.compiled, in); rc != 0 {
		compileMu.Lock()
		err := compileLastErr
		panicMsg := compileLastPanic
		compileLastErr = nil
		compileLastPanic = ""
		compileMu.Unlock()
		switch {
		case err != nil:
			return nil, fmt.Errorf("mlxgo: compiled function returned error: %w", err)
		case panicMsg != "":
			return nil, fmt.Errorf("mlxgo: compiled function panicked: %s", panicMsg)
		default:
			return nil, fmt.Errorf("mlxgo: compile apply failed (rc=%d)", int(rc))
		}
	}

	n := int(C.mlx_vector_array_size(out))
	results := make([]Array, n)
	for i := 0; i < n; i++ {
		var hndl C.mlx_array = C.mlx_array_new()
		if C.mlx_vector_array_get(&hndl, out, C.size_t(i)) != 0 {
			C.mlx_array_free(hndl)
			return nil, fmt.Errorf("mlxgo: compile apply: read output %d failed", i)
		}
		results[i] = wrap(hndl)
	}
	return results, nil
}
