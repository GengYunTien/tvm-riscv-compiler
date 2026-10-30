#!/usr/bin/env bash
# Apply the RVV BYOC backend onto the pinned TVM submodule.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TVM="$ROOT/3rdparty/tvm"
PATCH="$ROOT/patches/0001-add-rvv-cmake-options.patch"

cd "$TVM"
if git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH"
elif git apply --check -R "$PATCH" 2>/dev/null; then
  echo "CMake patch already applied; skipping"
else
  echo "ERROR: CMake patch neither applies nor is already applied" >&2
  exit 1
fi

mkdir -p src/relax/backend/contrib/rvv src/runtime/extra/contrib/rvv
cp "$ROOT/byoc/codegen.cc"          src/relax/backend/contrib/rvv/
cp "$ROOT/byoc/rvv_runtime.cc"      src/runtime/extra/contrib/rvv/
cp "$ROOT/cmake/RVV.cmake"          cmake/modules/contrib/
echo "RVV backend applied to $TVM"
