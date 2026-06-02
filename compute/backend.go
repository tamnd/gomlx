// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"fmt"

	"github.com/tamnd/gomlx/mlxgo"
)

// Backend is the device-facing half of the compute package. It owns the MLX
// stream and is the entry point a model forward pass calls into. The numeric
// helpers in this package (sampler, logits processors, cache bookkeeping,
// safetensors loading) work without it; Backend is only needed once tensors
// must run on the GPU.
type Backend struct {
	stream mlxgo.Stream
}

// Available reports whether this binary was built with a linked MLX runtime
// (the "mlx" build tag). When false, NewBackend fails and the server falls back
// to the mock engine instead of attempting real inference.
func Available() bool { return mlxgo.Available }

// NewBackend acquires the default GPU stream. It returns a descriptive error on
// a stub build so the caller can surface a clear "rebuild with -tags mlx"
// message rather than a generic nil-pointer failure deep in a forward pass.
func NewBackend() (*Backend, error) {
	if !mlxgo.Available {
		return nil, fmt.Errorf("compute: %w", mlxgo.ErrUnavailable)
	}
	s, err := mlxgo.NewStream()
	if err != nil {
		return nil, fmt.Errorf("compute: open MLX stream: %w", err)
	}
	return &Backend{stream: s}, nil
}
