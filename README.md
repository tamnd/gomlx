# gomlx

A Go reimplementation of [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX): an
OpenAI/Anthropic-compatible local LLM inference server for Apple Silicon, built on Apple's MLX via
its C API.

The goal is full feature parity with rapid-mlx and a 2x improvement in serving overhead and
concurrent throughput. Single-stream tokens/sec is bounded by the same Metal GPU kernels rapid-mlx
uses (called through cgo), so gomlx aims to match it; the win comes from Go's runtime — no GIL, no
asyncio, goroutine-per-request I/O, zero-alloc SSE, and CPU work kept off the GPU step thread.

## Status

Early development. The serving and parser layers (pure Go) come first and are independently
testable; the compute backend (cgo to `mlx-c`, reimplementing model forwards/caches/samplers in Go)
is the long pole and lands after.

Build order and design live in the spec set (`01`–`12`).

## Build

```
go build ./...
go test ./...
```

## Usage

```
gomlx models                  # list known model aliases
gomlx serve qwen3.5-4b        # start the server (serving layer wiring in progress)
```

## License

Apache-2.0.
