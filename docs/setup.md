# Setup Guide

## Host Machine Information

- Ubuntu 22.04 x86-64

## 1. Create conda environment

```bash
conda create -n tvm-build-venv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11
conda activate tvm-build-venv
```

## 2. Build TVM from Source

```bash
git clone --recursive https://github.com/apache/tvm
cd tvm

rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .

echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
echo "set(USE_CUDA   OFF)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake
echo "set(USE_CUBLAS OFF)" >> config.cmake
echo "set(USE_CUDNN  OFF)" >> config.cmake
echo "set(USE_CUTLASS OFF)" >> config.cmake

cmake .. && cmake --build . --parallel $(nproc)
```

## 3. Install tvm-ffi

```bash
cd ../3rdparty/tvm-ffi
pip install .
```

## 4. Install Python Dependencies

```bash
cd /path/to/tvm-riscv-compiler
pip install -r requirements.txt
```

## 5. Set Environment Variables

```bash
echo 'export TVM_HOME=/path/to/tvm' >> ~/.bashrc
echo 'export TVM_LIBRARY_PATH=$TVM_HOME/build' >> ~/.bashrc
echo 'export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## 6. Validate Installation
```bash
python3 -c "import tvm; print(tvm.__file__)"
```