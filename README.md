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
├── models/yolov8n/ # Model export
├── scripts/ # compile, validate
├── cmake/ # Cross-compile toolchain
├── runtime/ # Inference runtime
└── docs/ # Build and setup guides
```

## Quick Start

1. Set up development environment → [docs/setup.md](docs/setup.md)
2. Build TVM runtime for RISC-V → [docs/build-riscv-runtime.md](docs/build-riscv-runtime.md)
3. Build inference runtime → [docs/build-inference-runtime.md](docs/build-inference-runtime.md)

### Compile model

```bash
# x86 validation
python scripts/compile.py --target x86

# Cross-compile for RISC-V
python scripts/compile.py --target riscv-scalar
python scripts/compile.py --target riscv-rvv
```

### Deploy and run on BPI-F3

```bash
# Transfer files
scp yolov8n_riscv_rvv.so runtime/build/inference root@<board-ip>:/path/to/your/workspace

# Run on board
./inference /path/to/your/workspace/yolov8n_riscv_rvv.so
```

## Benchmark Results

> Zero pipeline baseline on BananaPi BPI-F3 (SpacemiT K1)
> Input: YOLOv8n, (1, 3, 640, 640) float32, single run

| Target | Pipeline | Inference Time |
|--------|----------|----------------|
| RISC-V scalar | zero | 42626 ms |
| RISC-V RVV | zero | 42612 ms |

**Note:** Both targets show similar performance under `zero` pipeline because
LLVM auto-vectorization does not generate RVV instructions for TVM's conv2d
kernels. The next step is to apply Relax optimization passes and implement
BYOC with hand-written RVV kernels.

## Roadmap

- [x] x86 baseline: compile + validate
- [x] RISC-V scalar baseline: 42626 ms
- [x] RISC-V RVV baseline: 42612 ms
- [ ] Relax optimization pipeline
- [ ] BYOC pattern registration
- [ ] Hand-written RVV kernels
- [ ] End-to-end benchmark comparison