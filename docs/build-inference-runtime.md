# Build Inference Runtime

## Prerequisites

Cross-compiler and TVM runtime already built.
See [build-riscv-runtime.md](build-riscv-runtime.md).

## Steps

```bash
cd runtime
mkdir build && cd build

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../../cmake/riscv64-toolchain.cmake \
    -DTVM_HOME=/path/to/tvm

make -j$(nproc)
```

## Verify

```bash
file build/inference
# Expected: ELF 64-bit LSB pie executable, UCB RISC-V
```