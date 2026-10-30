/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/extra/contrib/rvv/rvv_runtime.cc
 * \brief RVV BYOC runtime. Kernels are loaded at run time from a shared
 *        library (env RVV_KERNEL_LIB, default librvv_kernels.so), so the
 *        kernel can be rebuilt without rebuilding TVM.
 */

#include <dlfcn.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

// Kernel ABI. Bump RVV_KERNEL_ABI_VERSION only if this signature changes.
constexpr int RVV_KERNEL_ABI_VERSION = 1;

typedef void (*Conv2DFn)(const float* input, const float* weight, float* output,
                         int N, int C_in, int H, int W,
                         int C_out, int kH, int kW,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w);
typedef int (*AbiVersionFn)(void);
typedef const char* (*NameFn)(void);

/*! \brief Process-wide kernel library, loaded once. */
struct KernelLib {
  Conv2DFn conv2d = nullptr;
  std::string name;

  static KernelLib& Get() {
    static KernelLib lib;
    static std::once_flag flag;
    std::call_once(flag, [&]() { lib.Load(); });
    return lib;
  }

 private:
  void Load() {
    const char* env = std::getenv("RVV_KERNEL_LIB");
    std::string path = env ? env : "librvv_kernels.so";
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    TVM_FFI_ICHECK(handle != nullptr)
        << "RVV: cannot load kernel library '" << path << "': " << dlerror()
        << " (set RVV_KERNEL_LIB to its path)";

    auto abi = reinterpret_cast<AbiVersionFn>(dlsym(handle, "rvv_kernel_abi_version"));
    TVM_FFI_ICHECK(abi != nullptr) << "RVV: kernel library lacks rvv_kernel_abi_version";
    TVM_FFI_ICHECK_EQ(abi(), RVV_KERNEL_ABI_VERSION)
        << "RVV: kernel ABI version mismatch (library " << abi() << ", runtime "
        << RVV_KERNEL_ABI_VERSION << ")";

    conv2d = reinterpret_cast<Conv2DFn>(dlsym(handle, "conv2d_rvv"));
    TVM_FFI_ICHECK(conv2d != nullptr) << "RVV: kernel library lacks conv2d_rvv";

    auto name_fn = reinterpret_cast<NameFn>(dlsym(handle, "rvv_kernel_name"));
    name = name_fn ? name_fn() : "unknown";
    LOG(INFO) << "RVV kernel library: " << path << " (kernel: " << name << ")";
  }
};

class RVVRuntime : public JSONRuntimeBase {
 public:
  RVVRuntime(const std::string& symbol_name, const std::string& graph_json,
             const ffi::Array<ffi::String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  ~RVVRuntime() = default;

  const char* kind() const override { return "rvv_json"; }

  void Init(const ffi::Array<Tensor>& consts) override {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";
    SetupConstants(consts);
    KernelLib::Get();  // fail early if the kernel library is missing
  }

  void Run() override {
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() != "kernel") continue;
      const std::string& op_name = node.GetOpName();
      if (op_name.find("conv2d") != std::string::npos) {
        ExecuteConv2D(node, nid);
      } else {
        TVM_FFI_THROW(InternalError) << "RVV: unsupported op: " << op_name;
      }
    }
  }

 private:
  static int ToInt(int64_t v) { return static_cast<int>(v); }

  void ExecuteConv2D(const JSONGraphNode& node, uint32_t nid) {
    auto inputs = node.GetInputs();
    const DLTensor* in_t = data_entry_[EntryID(inputs[0])];
    const DLTensor* wt_t = data_entry_[EntryID(inputs[1])];
    const DLTensor* out_t = data_entry_[EntryID(nid, 0)];

    // Supported configuration checks: fail loudly instead of computing wrongly.
    TVM_FFI_ICHECK(in_t->dtype.code == kDLFloat && in_t->dtype.bits == 32)
        << "RVV conv2d: only float32 is supported";
    TVM_FFI_ICHECK(wt_t->dtype.code == kDLFloat && wt_t->dtype.bits == 32)
        << "RVV conv2d: only float32 weights are supported";
    TVM_FFI_ICHECK_EQ(in_t->ndim, 4) << "RVV conv2d: input must be 4-D (NCHW)";
    TVM_FFI_ICHECK_EQ(wt_t->ndim, 4) << "RVV conv2d: weight must be 4-D (OIHW)";

    auto strides = node.GetAttr<ffi::Array<int64_t>>("strides");
    auto padding = node.GetAttr<ffi::Array<int64_t>>("padding");
    auto dilation = node.GetAttr<ffi::Array<int64_t>>("dilation");
    int64_t groups = node.GetAttr<int64_t>("groups");

    TVM_FFI_ICHECK_EQ(groups, 1) << "RVV conv2d: groups != 1 is not supported";
    TVM_FFI_ICHECK(dilation.size() == 2 && dilation[0] == 1 && dilation[1] == 1)
        << "RVV conv2d: dilation != 1 is not supported";
    TVM_FFI_ICHECK(strides.size() == 2) << "RVV conv2d: expected 2 stride values";
    TVM_FFI_ICHECK(padding.size() == 4 && padding[0] == padding[2] && padding[1] == padding[3])
        << "RVV conv2d: only symmetric padding is supported";

    int N = ToInt(in_t->shape[0]), C_in = ToInt(in_t->shape[1]);
    int H = ToInt(in_t->shape[2]), W = ToInt(in_t->shape[3]);
    int C_out = ToInt(wt_t->shape[0]);
    int kH = ToInt(wt_t->shape[2]), kW = ToInt(wt_t->shape[3]);
    TVM_FFI_ICHECK_EQ(ToInt(wt_t->shape[1]), C_in) << "RVV conv2d: channel mismatch";

    KernelLib::Get().conv2d(static_cast<const float*>(in_t->data),
                            static_cast<const float*>(wt_t->data),
                            static_cast<float*>(out_t->data), N, C_in, H, W, C_out, kH, kW,
                            ToInt(strides[0]), ToInt(strides[1]), ToInt(padding[0]),
                            ToInt(padding[1]));
  }
};

ffi::Module RVVRuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                             const ffi::Array<ffi::String>& const_names) {
  auto n = tvm::ffi::make_object<RVVRuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.RVVJSONRuntimeCreate", RVVRuntimeCreate)
      .def("ffi.Module.load_from_bytes.rvv_json", JSONRuntimeBase::LoadFromBytes<RVVRuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm