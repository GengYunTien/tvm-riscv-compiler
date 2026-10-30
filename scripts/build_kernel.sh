#!/usr/bin/env bash
# Build librvv_kernels.so from conv2d_rvv.c.
# Same -O2 everywhere so timings are comparable.
#   x86  : run on the host   src: <repo>/kernels/rvv/conv2d_rvv.c  -> <repo>/librvv_kernels.so
#   riscv: run ON the BPI-F3 src: /usr/share/workspace/conv2d_rvv.c -> /usr/share/workspace/librvv_kernels.so
set -euo pipefail
MODE="${1:-}"
CFLAGS="-O2 -fPIC -shared"

case "$MODE" in
  x86)
    ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    SRC="$ROOT/kernels/rvv/conv2d_rvv.c"
    OUT="$ROOT/librvv_kernels.so"
    gcc $CFLAGS "$SRC" -o "$OUT"
    ;;
  riscv)
    [ "$(uname -m)" = "riscv64" ] || { echo "riscv mode must run on the riscv64 board" >&2; exit 1; }
    SRC="/usr/share/workspace/conv2d_rvv.c"
    OUT="/usr/share/workspace/librvv_kernels.so"
    gcc $CFLAGS -march=rv64gcv "$SRC" -o "$OUT"
    ;;
  *)
    echo "usage: $0 {x86|riscv}" >&2; exit 1 ;;
esac
echo "built $OUT ($(gcc --version | head -1))"