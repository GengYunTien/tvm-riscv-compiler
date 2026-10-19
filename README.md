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
├── tir/ # TensorIR kernel schedules
├── kernels/rvv/ # RVV intrinsic kernels
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

| Pipeline | scalar | rvv |
|----------|--------|-----|
| zero | 42626 ms | 42612 ms |
| fusion | 42669 ms | 42556 ms |
| fusion + BN fold | 42483 ms | 42714 ms |
| NHWC + tiling + vectorize | 14320 ms | 44890 ms |

**Analysis:**
- `zero` pipeline: LLVM auto-vectorization generates minimal RVV instructions for Conv2d kernels — scalar and RVV show similar performance
- `fusion` pipeline: FuseOps fuses bn+sigmoid+multiply into single kernels, but minimal speedup
- `fusion + BN fold`: FoldBatchnormToConv2D folds all 57 BN ops into Conv2d weights, still minimal speedup
- `NHWC + tiling + vectorize`: ConvertLayout (NCHW→NHWC) + TensorIR split/reorder/vectorize on ff axis — scalar -66% (42626ms→14320ms), but RVV slower than scalar due to LLVM selecting m2 with vsetvli overhead vs scalar loop unrolling
- Bottleneck confirmed: Conv2d computation dominates (~90%+ of inference time)
- Relax graph-level passes provide <1% improvement — TensorIR schedule and BYOC with hand-written RVV kernels required
- LLVM cannot effectively generate RVV code — hand-written RVV intrinsic (BYOC) required

## Roadmap

- [x] x86 baseline: compile + validate
- [x] RISC-V scalar/RVV baseline (zero pipeline): ~42600 ms
- [x] Relax optimization pipeline (fusion + BN fold): ~42500 ms
- [x] TensorIR schedule (NHWC tiling + vectorize): scalar 14320 ms (-66%)
- [ ] BYOC pattern registration
- [ ] Hand-written RVV kernels
- [ ] End-to-end benchmark comparison