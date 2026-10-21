cmake_minimum_required(VERSION 3.18)
project(run_model CXX)

set(CMAKE_CXX_STANDARD 17)

# TVM paths 
set(TVM_HOME "" CACHE PATH "Path to TVM source directory")
if(TVM_HOME STREQUAL "")
    message(FATAL_ERROR "Please set TVM_HOME: cmake -DTVM_HOME=/path/to/tvm ...")
endif()

# Include paths
include_directories(
    ${TVM_HOME}/include
    ${TVM_HOME}/3rdparty/tvm-ffi/include
    ${TVM_HOME}/3rdparty/tvm-ffi/3rdparty/dlpack/include
)

# Link paths
link_directories(${TVM_HOME}/build-riscv/lib)

# Executable
add_executable(run_model main.cc)

target_link_libraries(run_model
    -Wl,--no-as-needed
    tvm_runtime_extra
    -Wl,--as-needed
    tvm_runtime
    tvm_ffi
    pthread
    dl
)