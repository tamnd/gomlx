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
gomlx agents                  # list agent profiles
gomlx agents hermes           # show how to point one agent at the server
```

## Agents

Many coding agents speak the OpenAI API but each wants its config in a different shape, points at a
slightly different URL, and has its own streaming quirks. gomlx ships a profile for each known agent
describing what it needs, so configuring one is a lookup rather than guesswork.

```
gomlx agents                                  # list the profiles, most popular first
gomlx agents hermes -model qwen3.5-9b         # print a ready-to-use config for one
```

A profile carries the config format and template, the recommended models, the tool parser to force
if any, streaming tags the agent wraps around tool calls, and known issues. Profiles are versioned:
when an agent changes its config format, a version block overrides the base without breaking older
releases.

## MCP

gomlx can connect to Model Context Protocol servers and pool their tools. Point it at a JSON config
with `-mcp-config` and it dials every enabled server at startup, over a subprocess (stdio) or an HTTP
event stream (sse). A server that fails to connect is reported in its status rather than stopping the
others.

```
gomlx serve qwen3.5-4b -mcp-config mcp.json
```

```json
{
  "servers": {
    "files": {
      "transport": "stdio",
      "command": "my-files-mcp",
      "args": ["--root", "/tmp"]
    },
    "web": {
      "transport": "sse",
      "url": "http://127.0.0.1:9001/sse"
    }
  }
}
```

Three endpoints expose the subsystem: `GET /v1/mcp/tools` lists the pooled tools by their namespaced
`server__tool` name, `GET /v1/mcp/servers` reports each server's connection state, and
`POST /v1/mcp/execute` runs one tool by name. High-risk tools and dangerous arguments are refused at
the gate before a server is contacted.

## Embeddings

`POST /v1/embeddings` follows the OpenAI contract. It accepts all four input shapes: a single string,
a list of strings, a single pre-tokenized input (a list of integer token ids), and a batch of
pre-tokenized inputs (a list of those lists). The integer forms are kept off the string path, since
the token id 123 embeds differently from the word "123".

`dimensions` truncates each vector and renormalizes it to unit length, so a shortened vector stays
valid for cosine similarity. `encoding_format` is `float` (a JSON array) or `base64` (little-endian
float32 bytes), the latter saving bandwidth on large batches. Usage reports prompt and total tokens,
which are equal because embeddings have no completion side.

The embedding backend ships with the compute backend. Until then the endpoint reports itself
unconfigured with 503 rather than returning fabricated vectors.

## Benchmarks

Measured on an Apple M4 (24 GB) running Qwen3-0.6B in bf16, against a Python MLX serving baseline on
the same machine and weights with its prefix cache disabled so both do the same compute. The client
is `gomlx bench` driving the HTTP API. Numbers vary a few percent run to run.

| Workload | gomlx | Baseline | gomlx vs baseline |
| --- | --- | --- | --- |
| Serving overhead: 1 token, 16 concurrent, 128 requests | 157.7 req/s | 17.2 req/s | 9.2x |
| Short replies: 32 tokens, 16 concurrent, 64 requests | 326.9 tok/s, 10.2 req/s | 273.0 tok/s, 8.5 req/s | 1.20x |
| Sustained decode: 128 tokens, 8 concurrent, 32 requests | 188.7 tok/s | 275.3 tok/s | 0.69x |
| Single stream: 128 tokens, 1 client | 62.3 tok/s | 69.7 tok/s | 0.89x |

The serving overhead is where Go pays off. With one-token replies the time is almost all request
handling rather than GPU work, and gomlx serves these more than nine times faster: no interpreter
lock, a goroutine per request, and tokenize, detokenize, and JSON kept off the thread that drives the
GPU. That advantage still leads under a realistic mix of short replies at high concurrency, where
gomlx is about 1.2x ahead on both tokens per second and requests per second.

It trails on long, decode-bound runs. Single-stream throughput is set by the same Metal kernels for
both, and there gomlx tracks the hardware within about ten percent. Under sustained 128-token decode
the baseline pulls ahead because it compiles its model forward into one fused graph per step, while
gomlx still issues each MLX operation eagerly across the cgo boundary. Closing that gap means adding
graph compilation to the binding so a decode step is a single dispatch rather than dozens. That work
is tracked separately; the numbers above are reported as measured, wins and losses both.

## License

Apache-2.0.
