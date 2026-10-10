# Build TVM Runtime for RISC-V (BPI-F3)

## Prerequisites

```bash
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

## Steps

```bash
cd /path/to/tvm
mkdir build-riscv && cd build-riscv
cp ../cmake/config.cmake .

# Fix atomic linking issue
echo 'set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed -latomic")' >> config.cmake

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/tvm-riscv-compiler/cmake/riscv64-toolchain.cmake \
    -DUSE_LLVM=OFF \
    -DUSE_RPC=ON \
    -DBUILD_STATIC_RUNTIME=OFF

make runtime -j$(nproc)
```

## Verify

```bash
file lib/libtvm_runtime.so
# Expected: ELF 64-bit LSB shared object, UCB RISC-V
```

## Deploy to BPI-F3

### 1. Transfer libraries to board

```bash
scp lib/libtvm_runtime.so lib/libtvm_ffi.so root@<board-ip>:/usr/local/lib
```

### 2. On BPI-F3: install libatomic

```bash
apt install libatomic1
```

### 3. On BPI-F3: update library cache

```bash
ldconfig
```

### 4. On BPI-F3: verify

```bash
ldconfig -p | grep tvm
# Expected:
# libtvm_runtime.so (libc6,double-float) => /usr/local/lib/libtvm_runtime.so
# libtvm_ffi.so (libc6,double-float) => /usr/local/lib/libtvm_ffi.so
```