// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"fmt"

	"github.com/tamnd/gomlx/mlxgo"
)

// dtypeToMLX maps a safetensors dtype to the binding dtype. Only the types a
// model checkpoint actually stores are supported: the float types for weights
// and the unsigned 32-bit type for packed quantized weights.
func dtypeToMLX(d Dtype) (mlxgo.DType, error) {
	switch d {
	case F32:
		return mlxgo.F32, nil
	case F16:
		return mlxgo.F16, nil
	case BF16:
		return mlxgo.BF16, nil
	case U32:
		return mlxgo.U32, nil
	case I32:
		return mlxgo.I32, nil
	default:
		return 0, fmt.Errorf("compute: unsupported weight dtype %q", d)
	}
}

// LoadArray uploads one tensor from an open safetensors file to the device,
// preserving its dtype and shape. The byte view is copied into the MLX array,
// so it stays valid after the file is closed.
func LoadArray(st *SafeTensors, name string) (mlxgo.Array, error) {
	ti, ok := st.Header.Tensors[name]
	if !ok {
		return mlxgo.Array{}, fmt.Errorf("compute: tensor %q not found", name)
	}
	dt, err := dtypeToMLX(ti.Dtype)
	if err != nil {
		return mlxgo.Array{}, fmt.Errorf("compute: %q: %w", name, err)
	}
	raw, err := st.Bytes(name)
	if err != nil {
		return mlxgo.Array{}, err
	}
	return mlxgo.FromRawBytes(ti.Shape, dt, raw)
}
