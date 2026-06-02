# gomlx

gomlx is a local LLM inference server for Apple Silicon. It speaks the OpenAI and Anthropic HTTP
APIs and runs models on Apple's MLX through the MLX C API.

The aim is a serving layer that is fast and small. Single-stream throughput is set by the Metal GPU
kernels, which gomlx calls through cgo, so on that axis it tracks the hardware. Where Go helps is
everything around the model step: no global interpreter lock, a goroutine per request, streaming
responses with very little allocation, and tokenization, detokenization, and parsing kept off the
thread that drives the GPU. The result is lower per-request overhead and higher throughput when many
requests run at once.

## Status

Early development. The serving layer and the tool and reasoning parsers are written in pure Go and
can be tested on their own, so they come first. The compute backend (cgo to `mlx-c`, with the model
forward passes, KV caches, and samplers reimplemented in Go) is the hard part and lands after.

## Build

The default build is pure Go and runs everywhere, including Linux CI. It uses a mock decode backend,
so the full HTTP path, the parsers, and the numeric core (sampler, logits processors, cache
bookkeeping, safetensors loading) all build and test without a GPU.

```
go build ./...        # or: make build
go test ./...         # or: make test
```

### GPU backend (Apple Silicon)

Real inference links the MLX C API through cgo, which only compiles on Apple Silicon with an MLX
runtime present. A helper builds and installs that runtime into `third_party/mlx-c`, the path the
cgo binding expects:

```
make mlx-deps         # clone and build mlx-c into third_party/mlx-c
make build-mlx        # go build -tags mlx ./...
```

Without the `mlx` tag the backend reports itself unavailable and the server falls back to the mock
engine, so a missing MLX never breaks the build. Building MLX from source needs several gigabytes of
free disk; `scripts/bootstrap_mlx.sh` warns when space is tight and honors `MLX_C_REPO` and
`MLX_C_REF` to point at a pinned or prebuilt release.

## Usage

```
gomlx models                  # list known model aliases
gomlx serve qwen3.5-4b        # start the server (serving layer wiring in progress)
```

## License

Apache-2.0.
