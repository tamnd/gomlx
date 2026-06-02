#!/usr/bin/env bash
# Build the MLX C API locally so the GPU backend can link against it.
#
# This clones the MLX C bindings, builds them with CMake, and installs the
# headers and shared libraries into third_party/mlx-c, which is exactly where
# the cgo directives in mlxgo/mlx.go look (-I third_party/mlx-c/include and
# -L third_party/mlx-c/lib). After this finishes you can build the GPU path
# with: go build -tags mlx ./...
#
# Requirements: macOS on Apple Silicon, Xcode command line tools, cmake, and
# git. The build pulls in MLX itself as a CMake dependency and compiles Metal
# kernels, so it needs several gigabytes of free disk and a few minutes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/third_party/mlx-c"
BUILD_DIR="${REPO_ROOT}/third_party/mlx-c-build"
MLX_C_REPO="${MLX_C_REPO:-https://github.com/ml-explore/mlx-c.git}"
MLX_C_REF="${MLX_C_REF:-v0.2.0}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This backend targets Apple Silicon (arm64 macOS). Detected $(uname -s)/$(uname -m)." >&2
  echo "The default build (no mlx tag) works everywhere; only the GPU path needs this." >&2
  exit 1
fi

for tool in git cmake; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Missing required tool: ${tool}" >&2
    exit 1
  fi
done

avail_gb="$(df -g "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
if [[ -n "${avail_gb}" && "${avail_gb}" -lt 6 ]]; then
  echo "Warning: only ${avail_gb} GiB free; the MLX build needs roughly 6 GiB." >&2
  echo "Free some space or set MLX_C_REF to a prebuilt release before continuing." >&2
fi

SRC_DIR="${REPO_ROOT}/third_party/mlx-c-src"
if [[ ! -d "${SRC_DIR}/.git" ]]; then
  echo "Cloning mlx-c ${MLX_C_REF}..."
  git clone --depth 1 --branch "${MLX_C_REF}" "${MLX_C_REPO}" "${SRC_DIR}"
else
  echo "Reusing existing checkout at ${SRC_DIR}"
fi

echo "Configuring..."
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${DEST}" \
  -DMLX_C_BUILD_EXAMPLES=OFF

echo "Building..."
cmake --build "${BUILD_DIR}" --config Release -j"$(sysctl -n hw.ncpu)"

echo "Installing into ${DEST}..."
cmake --install "${BUILD_DIR}"

echo
echo "Done. Build the GPU backend with:"
echo "  go build -tags mlx ./..."
