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
| TVM | `4d3f3dacf2`, pinned as submodule, built from source |
| LLVM | 23.1.3 |
| Cross-compiler | riscv64-linux-gnu-gcc 11.4.0 |

## Project Structure

```
tvm-riscv-compiler/
├── 3rdparty/tvm/       # TVM (git submodule, pinned)
├── byoc/               # BYOC pattern registration, JSON codegen, runtime
├── kernels/rvv/        # RVV kernels (conv2d currently scalar placeholder)
├── patches/            # Patches applied to the TVM submodule
├── cmake/              # RVV build module, TVM config, cross-compile toolchain
├── scripts/            # TVM build and RVV backend apply scripts
├── models/yolov8n/     # Model export
├── inference/          # On-board model runner (run_model)
├── docs/               # Build and setup guides
├── compile.py          # Compile entry point (x86 also writes in.bin / ref_out.bin)
├── verify.py           # Compare board output with x86 reference
└── env.sh              # Environment variables (source before use)
```

## Quick Start

1. Clone with submodules and set up the environment → [docs/setup.md](docs/setup.md)
2. Build the riscv64 TVM runtime (with the RVV runtime) → [docs/build-riscv-runtime.md](docs/build-riscv-runtime.md)
3. Build the on-board inference executable → [docs/build-inference-runtime.md](docs/build-inference-runtime.md)

### Compile model

```bash
source env.sh

# x86 validation (BYOC conv2d, checked against PyTorch)
# host: generates in.bin (input) and ref_out.bin (x86 reference)
python compile.py --target x86

# Cross-compile for RISC-V
python compile.py --target riscv-scalar   # LLVM baseline, no RVV, no BYOC
python compile.py --target riscv-rvv      # LLVM +v, conv2d offloaded to RVV BYOC
```

### Deploy and run on BPI-F3

Install the riscv64 runtime libraries on the board first
(see [build-riscv-runtime.md](docs/build-riscv-runtime.md)), then:

```bash
# Transfer files
scp yolov8n_riscv_rvv.so inference/build/run_model root@<board-ip>:/path/to/your/workspace

# Run on board
./run_model /path/to/your/workspace/yolov8n_riscv_rvv.so --runs 5
```

### Verify on board

```bash
# host -> board
scp in.bin root@<board-ip>:/path/to/your/workspace

# board
./run_model yolov8n_riscv_rvv.so --input in.bin --output out.bin

# host
scp root@<board-ip>:/path/to/your/workspace/out.bin .
python verify.py
```

## Benchmark Results

> BananaPi BPI-F3 (SpacemiT K1), input: YOLOv8n, (1, 3, 640, 640) float32, single run
> single thread on one core (`taskset -c 0`), one warm-up run excluded.

### LLVM baselines (no BYOC)

| Pipeline | scalar | rvv (`+v`) |
|----------|--------|-----------|
| zero | 42626 ms | 42612 ms |
| fusion | 42669 ms | 42556 ms |
| fusion + BN fold | 42483 ms | 42714 ms |

**Analysis:**
- `zero`: LLVM auto-vectorization generates minimal RVV instructions for Conv2d; scalar and RVV perform similarly.
- `fusion`: FuseOps fuses bn+sigmoid+multiply into single kernels, but speedup is minimal.
- `fusion + BN fold`: FoldBatchnormToConv2D folds all 57 BN ops into Conv2d weights, still minimal speedup.
- Conv2d dominates inference time (~90%+); Relax graph-level passes give <1%.
- TensorIR experiment (commit `9d3f068`, since removed): NCHW→NHWC + split/reorder/vectorize on the output-channel axis cut scalar to 14320 ms (-66%), but the RVV build got slower (44890 ms) because LLVM chose `m2` with `vsetvli` overhead, while scalar benefited from loop unrolling.
- Conclusion: LLVM cannot effectively generate RVV code here; hand-written RVV intrinsics via BYOC are required.

### BYOC (conv2d offloaded to the RVV runtime)

| Pipeline | conv2d | Time |
|----------|--------|------|
| riscv-scalar | LLVM | 42249 ms |
| riscv-rvv | BYOC scalar placeholder | 67983 ms |

The placeholder output is bit-identical to the LLVM build on the board and
within 5e-4 abs / 3.6e-5 rel of the x86 reference. It validates the BYOC path
end to end, not performance.

## Roadmap

- [x] x86 baseline: compile + validate
- [x] RISC-V scalar/RVV baselines (zero, fusion, BN fold): ~42500 ms
- [x] TensorIR schedule experiment: scalar 14320 ms (-66%), RVV slower; removed
- [x] BYOC pattern registration, JSON codegen and runtime (verified on x86, scalar conv2d placeholder)
- [x] Reproducible build: TVM pinned as submodule
- [x] riscv64 TVM runtime including the RVV runtime
- [x] On-board numerical verification (compare output with x86)
- [ ] Hand-written RVV conv2d kernel
- [ ] Reject asymmetric padding in the RVV runtime
- [ ] End-to-end benchmark on BPI-F3