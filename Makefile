# gomlx build targets.
#
# The default build is pure Go and runs everywhere, including CI on Linux. The
# GPU backend links MLX through cgo and only builds on Apple Silicon after
# `make mlx-deps` has installed the MLX C library into third_party/mlx-c.

GO ?= go
PKGS ?= ./...

.PHONY: build
build: ## Build the pure-Go binary (mock engine, no GPU)
	$(GO) build $(PKGS)

.PHONY: build-mlx
build-mlx: ## Build with the MLX GPU backend (Apple Silicon, needs mlx-deps)
	$(GO) build -tags mlx $(PKGS)

.PHONY: test
test: ## Run the test suite with the race detector
	$(GO) test -race -count=1 $(PKGS)

.PHONY: vet
vet: ## Run go vet
	$(GO) vet $(PKGS)

.PHONY: fmt
fmt: ## Format all Go source
	gofmt -w .

.PHONY: fmt-check
fmt-check: ## Fail if any file is not gofmt-clean
	@out="$$(gofmt -l .)"; \
	if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

.PHONY: check
check: fmt-check vet test ## Run the full local check (matches CI)

.PHONY: mlx-deps
mlx-deps: ## Build and install the MLX C library into third_party/mlx-c
	./scripts/bootstrap_mlx.sh

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
