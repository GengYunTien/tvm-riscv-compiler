# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# RVV Codegen
if(USE_RVV_CODEGEN)
    message(STATUS "Build with RVV codegen")

    tvm_file_glob(GLOB COMPILER_RVV_SRCS src/relax/backend/contrib/rvv/*.cc)
    list(APPEND COMPILER_SRCS ${COMPILER_RVV_SRCS})

    tvm_file_glob(GLOB RUNTIME_RVV_SRCS src/runtime/extra/contrib/rvv/*.cc)
    if(NOT USE_RVV_RUNTIME)
        list(APPEND COMPILER_SRCS ${RUNTIME_RVV_SRCS})
    endif()
endif()

# RVV Runtime: goes into libtvm_runtime_extra. The kernel itself is NOT built
# here; it is loaded with dlopen from librvv_kernels.so (see scripts/build_kernel.sh).
if(USE_RVV_RUNTIME)
    message(STATUS "Build with RVV runtime")

    tvm_file_glob(GLOB RUNTIME_RVV_SRCS src/runtime/extra/contrib/rvv/*.cc)
    add_library(tvm_rvv_objs OBJECT ${RUNTIME_RVV_SRCS})
    target_link_libraries(tvm_rvv_objs PRIVATE tvm_runtime_extra_defs)
    target_link_libraries(tvm_runtime_extra PRIVATE tvm_rvv_objs ${CMAKE_DL_LIBS})

    add_definitions(-DTVM_GRAPH_EXECUTOR_RVV)
endif()