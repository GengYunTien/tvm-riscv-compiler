#!/usr/bin/env bash
# Build librvv_kernels.so ON the BPI-F3 (gcc 13 supports RVV 1.0 intrinsics; the host cross-compiler gcc 11.4 does not).
set -euo pipefail

[ "$(uname -m)" = "riscv64" ] || { echo "must run on the riscv64 board" >&2; exit 1; }

DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/conv2d_rvv.c"
OUT="$DIR/librvv_kernels.so"
CFLAGS="-O2 -fPIC -shared -march=rv64gcv"

gcc $CFLAGS "$SRC" -o "$OUT"
echo "built $OUT ($(gcc --version | head -1))"