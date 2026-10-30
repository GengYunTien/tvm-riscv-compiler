# Setup Guide

## Host Machine Information

- Ubuntu 22.04 x86-64

## Verified Versions

This project is validated against the following versions. TVM is pinned and
must not be upgraded: the BYOC code depends on this commit's API (for example
`JSONRuntimeBase::Init(const ffi::Array<Tensor>&)`), and TVM `main` changes
quickly (TIRx split, `USE_*` -> `TVM_USE_*` option renames, tvm-ffi updates).

| Component | Version |
|-----------|---------|
| TVM | `4d3f3dacf2` (2026-09-29), pinned as submodule `3rdparty/tvm` |
| tvm-ffi | `c96f9e31b` (TVM submodule, installed from source) |
| LLVM | 23.1.3 (23.1.2 also verified) |
| cmake | 4.4.4 (4.4.3 also verified) |
| Python | 3.11 |
| Board gcc | 13.3 (builds the RVV kernel on the BPI-F3) |

The x86 TVM build is used only to run `compile.py`. Models are executed on the
board.

## 1. Clone with submodules

```bash
git clone --recursive https://github.com/GengYunTien/tvm-riscv-compiler
cd tvm-riscv-compiler
```

If already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## 2. Create conda environment

```bash
conda create -n tvm-rvv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11
conda activate tvm-rvv
```

The version constraints above resolve to the latest conda-forge release, which
may differ from the verified versions. Check them:

```bash
llvm-config --version
cmake --version | head -1
```

If the build fails with other versions, pin the verified ones.

## 3. Build TVM with the RVV BYOC backend

```bash
./scripts/build_tvm.sh
```

The script does the following:

1. `scripts/apply_rvv_backend.sh`: applies `patches/0001-add-rvv-cmake-options.patch`
   to the TVM submodule and copies the RVV backend files
   (`byoc/codegen.cc`, `byoc/rvv_runtime.cc`, `cmake/RVV.cmake`) into the TVM
   source tree. The kernel (`kernels/rvv/conv2d_rvv.c`) is not part of TVM; the
   runtime loads it at run time with `dlopen`.
2. Copies `cmake/tvm_config.cmake` to `3rdparty/tvm/build/config.cmake`.
3. Runs `cmake .. -DUSE_RVV_CODEGEN=ON -DUSE_RVV_RUNTIME=ON` and builds
   (a full build takes 30+ minutes).
4. Installs tvm-ffi from the submodule (`3rdparty/tvm/3rdparty/tvm-ffi`).

The configure step must print:

```
-- Build with RVV codegen
-- Build with RVV runtime
```

Note: the submodule working tree is modified after this step. This is expected.
Do not commit inside `3rdparty/tvm`.

## 4. Install Python dependencies

Run after step 3, so pip does not pull `apache-tvm-ffi` from PyPI:

```bash
pip install -r requirements.txt
pip show apache-tvm-ffi | grep -i version   # must contain +gc96f9e31b
```

## 5. Set environment variables

```bash
source env.sh
```

This sets `TVM_HOME`, `TVM_LIBRARY_PATH` and `PYTHONPATH` for the current shell
only. Run it again in every new terminal.

## 6. Validate

```bash
python3 -c "
import tvm, tvm_ffi
print('tvm:', tvm.__file__)
print('relax.ext.rvv:', 'OK' if tvm_ffi.get_global_func('relax.ext.rvv', True) else 'NOT FOUND')
print('runtime.RVVJSONRuntimeCreate:', 'OK' if tvm_ffi.get_global_func('runtime.RVVJSONRuntimeCreate', True) else 'NOT FOUND')
"
```

Both must print `OK`. Then compile the model both ways and check that only the
BYOC build contains the RVV module (one per conv2d, 41 for YOLOv8n):

```bash
python compile.py --conv2d llvm
python compile.py --conv2d byoc
strings yolov8n_riscv_llvm.so | grep -c rvv_json   # 0
strings yolov8n_riscv_byoc.so | grep -c rvv_json   # 41
```

Next steps: build the riscv64 runtime
([build-riscv-runtime.md](build-riscv-runtime.md)), the on-board runner
([build-inference-runtime.md](build-inference-runtime.md)), then follow
"Verify on board" in the README.

## Known limitations

- The RVV conv2d supports float32, NCHW, groups = 1 and dilation = 1. The
  runtime raises an error for anything else, including asymmetric padding.
- Only conv2d is offloaded to the RVV backend; other operators are compiled by
  LLVM without `+v`.
- The kernel must be built on the board (host gcc 11.4 has no
  `riscv_vector.h`), see the README ("Build the kernel").