#!/usr/bin/env bash
# Build the pinned TVM with the RVV BYOC backend, then install tvm-ffi.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TVM="$ROOT/3rdparty/tvm"

"$ROOT/scripts/apply_rvv_backend.sh"

mkdir -p "$TVM/build"
cd "$TVM/build"
cp "$ROOT/cmake/tvm_config.cmake" config.cmake
cmake .. -DUSE_RVV_CODEGEN=ON -DUSE_RVV_RUNTIME=ON
cmake --build . --parallel "$(nproc)"

cd "$TVM/3rdparty/tvm-ffi"
pip install .
