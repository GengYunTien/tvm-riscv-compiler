# Build Inference Runtime

## Prerequisites

Cross-compiler and the riscv64 TVM runtime (built **with** the RVV runtime,
`-DUSE_RVV_RUNTIME=ON`) must already be built.
See [build-riscv-runtime.md](build-riscv-runtime.md).

## Steps

```bash
cd inference
rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../../cmake/riscv64-toolchain.cmake \
    -DTVM_HOME=$(realpath ../../3rdparty/tvm)

make -j$(nproc)
```

## Verify

```bash
file run_model
# Expected: ELF 64-bit LSB pie executable, UCB RISC-V
```

Copy it to the board and check that it links the RVV runtime library:

```bash
scp run_model root@<board-ip>:/path/to/your/workspace
ssh root@<board-ip> "ldd /path/to/your/workspace/run_model | grep tvm"
# Expected: libtvm_runtime.so, libtvm_runtime_extra.so, libtvm_ffi.so
```

Without `libtvm_runtime_extra.so`, loading a model that uses the RVV BYOC
backend fails with "a loader of that name is not registered" (`rvv_json`).