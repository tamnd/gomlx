// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"testing"

	"github.com/tamnd/gomlx/mlxgo"
)

func TestDtypeToMLX(t *testing.T) {
	cases := map[Dtype]mlxgo.DType{
		F32: mlxgo.F32, F16: mlxgo.F16, BF16: mlxgo.BF16, U32: mlxgo.U32, I32: mlxgo.I32,
	}
	for d, want := range cases {
		got, err := dtypeToMLX(d)
		if err != nil || got != want {
			t.Errorf("%s: got %v err %v want %v", d, got, err, want)
		}
	}
	if _, err := dtypeToMLX(F64); err == nil {
		t.Error("F64 should be unsupported for weights")
	}
}

func TestLoadArrayMissingTensor(t *testing.T) {
	img := buildSafetensors(t, nil, []struct {
		name    string
		dtype   string
		shape   []int
		payload []byte
	}{
		{"w", "F32", []int{2}, f32bytes(1, 2)},
	})
	st, err := FromBytes(img)
	if err != nil {
		t.Fatalf("FromBytes: %v", err)
	}
	defer st.Close()
	if _, err := LoadArray(st, "nope"); err == nil {
		t.Error("expected error for missing tensor")
	}
}
