# tvm-riscv-compiler

TVM BYOC compiler targeting RISC-V RVV on BananaPi BPI-F3 (SpacemiT K1) —
from YOLOv8n import to vectorized inference.

## Hardware

| Component | Spec |
|-----------|------|
| Board | BananaPi BPI-F3 |
| SoC | SpacemiT K1 |
| CPU | 8-core RISC-V X60 |
| ISA | RV64GCV + RVA22 + RVV 1.0 (VLEN=256) |
| OS | Armbian Ubuntu 24.04 Noble |

## Host Machine

| Component | Spec |
|-----------|------|
| OS | Ubuntu 22.04 x86-64 |
| Python | 3.11 (conda) |
| TVM | built from source |
| Cross-compiler | riscv64-linux-gnu-gcc 11.4.0 |

## Project Structure

```
tvm-riscv-compiler/
├── byoc/ # BYOC pattern registration, partition, codegen
├── kernels/rvv/ # RVV intrinsic kernels
├── relax/ # Relax IR passes
├── models/yolov8n/ # Model export and pre/postprocess
├── scripts/ # compile, inference, benchmark
├── cmake/ # Cross-compile toolchain
├── runtime/ # C++ inference runtime
└── docs/ # Build and setup guides
```

## Quick Start

See [docs/setup.md](docs/setup.md) for full setup instructions.