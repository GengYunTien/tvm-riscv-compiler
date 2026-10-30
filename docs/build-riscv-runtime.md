# Build TVM Runtime for RISC-V (BPI-F3)

## Prerequisites

```bash
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

The TVM submodule must already have the RVV backend applied
(`./scripts/build_tvm.sh` runs `scripts/apply_rvv_backend.sh`).

## Steps

```bash
cd 3rdparty/tvm
mkdir build-riscv && cd build-riscv
cp ../cmake/config.cmake .

# Fix atomic linking issue
echo 'set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed -latomic")' >> config.cmake

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$(realpath ../../../cmake/riscv64-toolchain.cmake) \
    -DUSE_LLVM=OFF \
    -DUSE_RPC=ON \
    -DBUILD_STATIC_RUNTIME=OFF \
    -DUSE_RVV_RUNTIME=ON

make tvm_runtime tvm_runtime_extra tvm_ffi_shared -j$(nproc)
```

The configure step must print `Build with RVV runtime`.

## Verify

```bash
file lib/libtvm_runtime_extra.so
# Expected: ELF 64-bit LSB shared object, UCB RISC-V
nm lib/libtvm_runtime_extra.so | grep -c "RVVRuntime"     # > 0
nm lib/libtvm_runtime_extra.so | grep -c "conv2d_rvv"     # 0 (kernel is loaded with dlopen)
```

## Deploy to BPI-F3

### 1. Transfer libraries to board

```bash
scp lib/libtvm_runtime.so lib/libtvm_runtime_extra.so lib/libtvm_ffi.so \
    root@<board-ip>:/usr/local/lib
```

### 2. On BPI-F3: install libatomic and update library cache

```bash
apt install libatomic1
ldconfig
```

### 3. On BPI-F3: verify

```bash
ldconfig -p | grep tvm
# Expected: libtvm_runtime.so, libtvm_runtime_extra.so, libtvm_ffi.so
```

### 4. Kernel library

The runtime loads the conv2d kernel at run time with `dlopen`. If
`RVV_KERNEL_LIB` is not set, it looks for `librvv_kernels.so` in the system
library search path (`LD_LIBRARY_PATH`, `ldconfig` cache), not in the current
directory. Always set `RVV_KERNEL_LIB` to an absolute path:

```bash
RVV_KERNEL_LIB=/usr/share/workspace/librvv_kernels.so ./run_model ...
```

Build the kernel on the board, see "Build the kernel" in the README. Use
absolute paths for both the model and the kernel library.

`libtvm_runtime_extra.so` holds the RVV runtime (`rvv_json` loader). The
`run_model` executable must link it, see
[build-inference-runtime.md](build-inference-runtime.md).