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
 * \brief RVV RISC-V Vector Extension runtime for conv2d acceleration.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

extern "C" void conv2d_rvv(
    const float* input,
    const float* weight,
    float*       output,
    int N,
    int C_in,  int H,     int W,
    int C_out, int kH,    int kW,
    int stride_h, int stride_w,
    int pad_h,    int pad_w
);

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

/*!
 * \brief RVV runtime implementation with architectural concepts
 */
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
    LOG(INFO) << "RVV Runtime initialized, graph nodes: " << nodes_.size();
  }

  void Run() override {
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        const std::string& op_name = node.GetOpName();
        if (op_name.find("conv2d") != std::string::npos) {
          ExecuteConv2D(node, nid);
        } else {
          LOG(WARNING) << "RVV: unsupported op: " << op_name;
        }
      }
    }
  }

 private:
  void ExecuteConv2D(const JSONGraphNode& node, uint32_t nid) {
    // Get input and weight entries
    auto inputs = node.GetInputs();
    // inputs[0] = input, inputs[1] = weight
    const DLTensor* input_tensor  = data_entry_[EntryID(inputs[0])];
    const DLTensor* weight_tensor = data_entry_[EntryID(inputs[1])];

    // Output is node nid, output index 0
    const DLTensor* output_tensor = data_entry_[EntryID(nid, 0)];

    const float* input_ptr  = static_cast<const float*>(input_tensor->data);
    const float* weight_ptr = static_cast<const float*>(weight_tensor->data);
    float*       output_ptr = static_cast<float*>(output_tensor->data);

    // Get shapes
    int N    = input_tensor->shape[0];
    int C_in = input_tensor->shape[1];
    int H    = input_tensor->shape[2];
    int W    = input_tensor->shape[3];

    int C_out = weight_tensor->shape[0];
    int kH    = weight_tensor->shape[2];
    int kW    = weight_tensor->shape[3];

    // Get conv2d attrs
    auto strides = node.GetAttr<ffi::Array<int64_t>>("strides");
    auto padding = node.GetAttr<ffi::Array<int64_t>>("padding");

    int stride_h = strides.size() >= 1 ? static_cast<int>(strides[0]) : 1;
    int stride_w = strides.size() >= 2 ? static_cast<int>(strides[1]) : 1;
    int pad_h    = padding.size() >= 1 ? static_cast<int>(padding[0]) : 0;
    int pad_w    = padding.size() >= 2 ? static_cast<int>(padding[1]) : 0;

    conv2d_rvv(input_ptr, weight_ptr, output_ptr,
               N, C_in, H, W,
               C_out, kH, kW,
               stride_h, stride_w,
               pad_h, pad_w);
  }
};

/*!
 * \brief Create the RVV runtime module
 */
ffi::Module RVVRuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                            const ffi::Array<ffi::String>& const_names) {
  auto n = tvm::ffi::make_object<RVVRuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.RVVJSONRuntimeCreate", RVVRuntimeCreate)
      .def("ffi.Module.load_from_bytes.rvv_json",
           JSONRuntimeBase::LoadFromBytes<RVVRuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
