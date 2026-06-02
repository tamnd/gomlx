// SPDX-License-Identifier: Apache-2.0

//go:build !mlx

package compute

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/gomlx/mlxgo"
)

// On the default build the GPU path is not linked, so the backend must report
// itself unavailable and fail to construct with a message that tells the user
// how to enable it.
func TestBackendUnavailableWithoutMLXTag(t *testing.T) {
	if Available() {
		t.Fatal("Available() should be false without the mlx build tag")
	}
	_, err := NewBackend()
	if err == nil {
		t.Fatal("NewBackend should fail on a stub build")
	}
	if !errors.Is(err, mlxgo.ErrUnavailable) {
		t.Errorf("error should wrap mlxgo.ErrUnavailable: %v", err)
	}
	if !strings.Contains(err.Error(), "-tags mlx") {
		t.Errorf("error should tell the user to rebuild with -tags mlx: %v", err)
	}
}
