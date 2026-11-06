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
| Cross-compiler | riscv64-linux-gnu-gcc 11.4.0 (TVM runtime and model) |

The RVV kernel is built on the board with gcc 14, because the host
cross-compiler (gcc 11.4) has no `riscv_vector.h`.

## Project Structure

```
tvm-riscv-compiler/
├── 3rdparty/tvm/       # TVM (git submodule, pinned)
├── byoc/               # BYOC pattern registration, JSON codegen, runtime
├── kernels/rvv/        # conv2d kernel (RVV intrinsics + scalar fallback, loaded via dlopen)
├── tests/              # Kernel unit test
├── patches/            # Patches applied to the TVM submodule
├── cmake/              # RVV build module, TVM config, cross-compile toolchain
├── scripts/            # TVM build, RVV backend apply, kernel build
├── models/yolov8n/     # Model export
├── inference/          # On-board model runner (run_model)
├── docs/               # Build and setup guides
├── compile.py          # Cross-compile the model for RISC-V
├── reference.py        # Generate fixed input and PyTorch reference output
├── verify.py           # Compare board output with the PyTorch reference
└── env.sh              # Environment variables (source before use)
```

## Quick Start

1. Clone with submodules and set up the environment → [docs/setup.md](docs/setup.md)
2. Build the riscv64 TVM runtime (with the RVV runtime) → [docs/build-riscv-runtime.md](docs/build-riscv-runtime.md)
3. Build the on-board model runner → [docs/build-inference-runtime.md](docs/build-inference-runtime.md)

### Compile model

```bash
source env.sh

python compile.py --conv2d llvm   # conv2d compiled by LLVM (baseline)
python compile.py --conv2d byoc   # conv2d offloaded to the RVV runtime
```

The rest of the model is compiled by LLVM without `+v`, so the two outputs
differ only in the conv2d implementation.

### Build the kernel (on the board)

Prerequisite (once): gcc 14 (Ubuntu 24.04 ships gcc 13 by default):

```bash
apt install build-essential gcc-14
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
update-alternatives --set gcc /usr/bin/gcc-14
gcc --version | head -1      # gcc 14.x
```

```bash
# host -> board
scp kernels/rvv/conv2d_rvv.c scripts/build_kernel.sh root@<board-ip>:/path/to/your/workspace

# board
./build_kernel.sh        # builds librvv_kernels.so
```

`conv2d_rvv.c` and `build_kernel.sh` must be in the same directory on the
board; `librvv_kernels.so` is written next to them. The runtime log prints the
kernel version (`kernel: rvv-v1-ow-m1`), so every measurement can be traced to
the kernel that produced it.

### Deploy and run on BPI-F3

Install the riscv64 runtime libraries on the board first
(see [build-riscv-runtime.md](docs/build-riscv-runtime.md)), then:

```bash
# host -> board
scp yolov8n_riscv_byoc.so yolov8n_riscv_llvm.so inference/build/run_model \
    root@<board-ip>:/path/to/your/workspace

# board (use absolute paths)
taskset -c 0 ./run_model /path/to/your/workspace/yolov8n_riscv_llvm.so --runs 5
RVV_KERNEL_LIB=/path/to/your/workspace/librvv_kernels.so \
taskset -c 0 ./run_model /path/to/your/workspace/yolov8n_riscv_byoc.so --runs 5
```

### Verify on board

```bash
# host: fixed input and PyTorch reference
python reference.py                      # writes in.bin, ref_out.bin
scp in.bin root@<board-ip>:/path/to/your/workspace

# board
RVV_KERNEL_LIB=/path/to/your/workspace/librvv_kernels.so \
./run_model /path/to/your/workspace/yolov8n_riscv_byoc.so --input in.bin --output out.bin --runs 1

# host
scp root@<board-ip>:/path/to/your/workspace/out.bin .
python verify.py
```

### Kernel unit test

Runs in seconds and covers stride, padding, 1x1 and channel counts that are not
a multiple of the vector length. Build on the board:

```bash
# host -> board
scp tests/test_conv2d.c kernels/rvv/conv2d_rvv.c root@<board-ip>:/path/to/your/workspace

# board
gcc -O2 -march=rv64gcv -o test_conv2d test_conv2d.c conv2d_rvv.c -lm && ./test_conv2d
objdump -d test_conv2d | grep -c vsetvli     # > 0: the vector path was compiled
```

## Benchmark Results

> BananaPi BPI-F3 (SpacemiT K1), input: YOLOv8n, (1, 3, 640, 640) float32
> (`in.bin`), single thread on one core (`taskset -c 0`), one warm-up run
> excluded. Time is the median of 3 runs. Counters come from `perf stat` over
> the warm-up run plus one timed run. Kernels are built with gcc 14 `-O2`.

### Current

| conv2d | Time | Instructions | IPC | L1 load misses |
|--------|------|--------------|-----|----------------|
| LLVM | 42211 ms | 37.9 G | 0.28 | 1.44 G |
| scalar C kernel | 76496 ms | 200.0 G | 0.81 | 2.41 G |
| RVV v1 (`ow` vectorized, masked gather, m1) | 28362 ms | 25.1 G | 0.27 | 0.42 G |

All three match the PyTorch reference (max abs diff 8.2e-4, max rel diff 7.5e-5).
Time is the whole model; operators other than conv2d are compiled by LLVM
without `+v`.

**Vectorization (scalar C kernel vs v1).** Instructions drop 8.0x, matching the
8 lanes of e32m1, but time improves only 2.7x because IPC falls from 0.81 to
0.27. Time is instructions times IPC, so 8.0 x (0.27 / 0.81) accounts for the
gap. L1 load misses fall 5.7x.

**Against LLVM.** LLVM's scalar code already executes 5.3x fewer instructions
than the naive C kernel (loop restructuring and unrolling). v1 is 1.49x faster
than LLVM with 1.51x fewer instructions and the same IPC, so the lead comes from
the instruction count. Raising IPC is the target of the next versions.

### Earlier LLVM baselines (not core-pinned, with and without `+v`)

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

## Roadmap

- [x] RISC-V scalar/RVV baselines (zero, fusion, BN fold): ~42500 ms
- [x] TensorIR schedule experiment: scalar 14320 ms (-66%), RVV slower; removed
- [x] BYOC pattern registration, JSON codegen and runtime
- [x] Reproducible build: TVM pinned as submodule
- [x] riscv64 TVM runtime including the RVV runtime
- [x] Kernel loaded via dlopen, so kernel changes do not rebuild TVM
- [x] Runtime rejects unsupported conv2d (non-fp32, groups/dilation != 1, asymmetric padding)
- [x] On-board numerical verification against PyTorch
- [x] Kernel unit test
- [x] RVV conv2d v1: vectorize along `ow`, masked gather (1.49x faster than LLVM)
- [x] Scalar C kernel measured with gcc 14, vectorization effect isolated (2.7x)
- [ ] RVV conv2d v2: contiguous loads for in-bounds segments, gather only at the edges
- [ ] RVV conv2d v3+: several output channels per input load, multiple accumulators, LMUL and L1 tiling
- [ ] Multi-threaded conv2d
- [ ] Additional operators