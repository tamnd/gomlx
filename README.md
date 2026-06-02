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
