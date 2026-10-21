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
 * \file src/relax/backend/contrib/rvv/codegen.cc
 * \brief RVV JSON codegen implementation.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr_functor.h>

#include <memory>
#include <string>

#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONSerializer = backend::contrib::JSONSerializer;
using OpAttrExtractor = backend::contrib::OpAttrExtractor;
using backend::contrib::NodeEntries;

/*!
 * \brief Collect the conv2d call and its attributes (strides, padding, ...)
 *        from the body of a composite function.
 */
class CollectConv2DAttrs : public ExprVisitor {
 public:
  CollectConv2DAttrs() : node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const CallNode* call_node) final {
    if (const auto* op_node = call_node->op.as<OpNode>()) {
      if (op_node->name == "relax.nn.conv2d") {
        OpAttrExtractor extractor(node_);
        const ffi::Object* attr_obj = call_node->attrs.get();
        extractor.Extract(const_cast<ffi::Object*>(attr_obj));
        conv_call_ = call_node;
      }
    }
    ExprVisitor::VisitExpr_(call_node);
  }

  /*! \brief Temporary node that holds the extracted attributes. */
  std::shared_ptr<JSONGraphNode> node_;
  /*! \brief The conv2d call found in the composite body. */
  const CallNode* conv_call_ = nullptr;
};

class RVVJSONSerializer : public JSONSerializer {
 public:
  RVVJSONSerializer(ffi::Map<GenericConst, ffi::String> constant_names,
                    ffi::Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  NodeEntries VisitExpr_(const CallNode* call_node) final {
    const auto* fn_var = call_node->op.as<VarNode>();
    TVM_FFI_ICHECK(fn_var);
    const auto fn = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();
    TVM_FFI_ICHECK(fn.defined()) << "Expects the callee to be a function.";

    auto composite_opt = fn->GetAttr<ffi::String>(attr::kComposite);
    TVM_FFI_ICHECK(composite_opt.has_value()) << "Only composite functions are supported.";

    std::string composite_name = composite_opt.value();

    // Find the conv2d call inside the composite function and extract its attributes.
    CollectConv2DAttrs collector;
    collector.VisitExpr(fn->body);
    TVM_FFI_ICHECK(collector.conv_call_ != nullptr)
        << "RVV composite " << composite_name << " must contain a relax.nn.conv2d call";

    NodeEntries inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, "kernel", inputs, 1);
    // Transfer strides / padding / dilation / groups ... to the final node.
    node->CaptureAttrs(*collector.node_);
    return AddNode(node, ffi::GetRef<Expr>(call_node));
  }

 private:
  ffi::Map<Var, Expr> bindings_;
};

ffi::Array<ffi::Module> RVVCompiler(ffi::Array<Function> functions,
                                    ffi::Map<ffi::String, ffi::Any> /*unused*/,
                                    ffi::Map<GenericConst, ffi::String> constant_names) {
  ffi::Array<ffi::Module> compiled_functions;
  const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.RVVJSONRuntimeCreate");

  for (const auto& func : functions) {
    RVVJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto const_names = serializer.GetConstantNames();
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back(pf(func_name, graph_json, const_names).cast<ffi::Module>());
  }

  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.rvv", RVVCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm