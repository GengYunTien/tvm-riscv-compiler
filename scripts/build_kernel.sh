#!/usr/bin/env bash
# Build librvv_kernels.so ON the BPI-F3 (gcc 13 supports RVV 1.0 intrinsics; the host cross-compiler gcc 11.4 does not).
#   src: /usr/share/workspace/conv2d_rvv.c
#   out: /usr/share/workspace/librvv_kernels.so
set -euo pipefail

[ "$(uname -m)" = "riscv64" ] || { echo "must run on the riscv64 board" >&2; exit 1; }

SRC="/usr/share/workspace/conv2d_rvv.c"
OUT="/usr/share/workspace/librvv_kernels.so"
CFLAGS="-O2 -fPIC -shared -march=rv64gcv"

gcc $CFLAGS "$SRC" -o "$OUT"
echo "built $OUT ($(gcc --version | head -1))"