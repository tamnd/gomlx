// SPDX-License-Identifier: Apache-2.0

//go:build mlx

package compute

import (
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkBatchDecode measures sustained batched decode throughput, the metric
// to watch when working the decode path. It prefills a short prompt across a
// batch and then times one-token decode steps, reporting tokens per second
// across the whole batch. The model and runtime are required, so the benchmark
// skips when GOMLX_QWEN3_DIR points nowhere usable.
//
// Run with, for example:
//
//	GOMLX_QWEN3_DIR=$HOME/models/qwen3-0.6b \
//	DYLD_LIBRARY_PATH=third_party/mlx-c/lib \
//	go test -tags mlx -run x -bench BatchDecode -benchtime 200x ./compute/
//
// Throughput on this hardware is noisy run to run (a small model at a modest
// batch is dominated by GPU kernel time and thermal state), so compare medians
// across several interleaved runs rather than a single number.
func BenchmarkBatchDecode(b *testing.B) {
	dir := os.Getenv("GOMLX_QWEN3_DIR")
	if dir == "" {
		dir = filepath.Join(os.Getenv("HOME"), "models", "qwen3-0.6b")
	}
	cfgRaw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		b.Skipf("Qwen3 model not present at %s", dir)
	}
	args, err := LoadQwen3Args(cfgRaw)
	if err != nil {
		b.Fatalf("LoadQwen3Args: %v", err)
	}
	st, err := Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		b.Fatalf("open weights: %v", err)
	}
	defer st.Close()
	model, err := NewQwen3Model(st, args)
	if err != nil {
		b.Fatalf("NewQwen3Model: %v", err)
	}

	const batchSize = 8
	prompts := make([][]int32, batchSize)
	for i := range prompts {
		prompts[i] = []int32{9707, 11, 1879, 0} // a short fixed prompt
	}
	batch := model.NewBatch(batchSize)
	if _, err := batch.Prefill(prompts); err != nil {
		b.Fatalf("prefill: %v", err)
	}

	next := make([]int32, batchSize)
	for i := range next {
		next[i] = 13
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		logits, err := batch.Decode(next)
		if err != nil {
			b.Fatalf("decode: %v", err)
		}
		// Force the lazy graph to evaluate so each step is real work, mirroring
		// the runner which reads logits back every step.
		_ = logits
	}
	b.StopTimer()

	// tokens/sec across the whole batch is the figure that matters for serving.
	steps := float64(b.N)
	b.ReportMetric(steps*batchSize/b.Elapsed().Seconds(), "tok/s")
}
