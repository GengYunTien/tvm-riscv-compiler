import numpy as np
import os
import argparse
from ultralytics import YOLO

import torch
from torch import nn
from torch.export import export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tir.schedules import apply_conv2d_schedule

parser = argparse.ArgumentParser(description = "Compile YOLOv8n for RISC-V")
parser.add_argument(
    "--target", 
    choices = ["x86", "riscv-scalar", "riscv-rvv"], 
    default = "x86", 
    help = "Compilation target"
)
args = parser.parse_args()

# Import from PyTorch
torch_model = YOLO('yolov8n.pt').model.eval()
example_args = (torch.randn(1, 3, 640, 640),)

with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod = from_exported_program(
        exported_program,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=True,
    )

# Target configuration
if args.target == "x86":
    target = tvm.target.Target("llvm")
    output_file = "yolov8n_x86.so"
    cross_compile = False

elif args.target == "riscv-scalar":
    target = tvm.target.Target({
        "kind": "llvm",
        "mtriple": "riscv64-linux-gnu",
        "mcpu": "generic-rv64",
        "mabi": "lp64d",
        "mattr": ["+64bit", "+m", "+a", "+f", "+d", "+c"]
    })
    output_file = "yolov8n_riscv_scalar.so"
    cross_compile = True

elif args.target == "riscv-rvv":
    target = tvm.target.Target({
        "kind": "llvm",
        "mtriple": "riscv64-linux-gnu",
        "mcpu": "generic-rv64",
        "mabi": "lp64d",
        "mattr": ["+64bit", "+m", "+a", "+f", "+d", "+c", "+v"]
    })
    output_file = "yolov8n_riscv_rvv.so"
    cross_compile = True

# Compile
mod_compiled = tvm.ir.transform.Sequential([
    relax.transform.DecomposeOpsForInference(),
    relax.transform.FoldBatchnormToConv2D(),
    relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
    relax.transform.LegalizeOps(),
    relax.transform.AnnotateTIROpPattern(),
    relax.transform.FuseOps(),
    relax.transform.FuseTIR(),
])(mod)
mod_compiled = apply_conv2d_schedule(mod_compiled)
exec_module = tvm.compile(mod_compiled, target=target)

# Export
if cross_compile:
    exec_module.export_library(
        output_file, 
        cc = "riscv64-linux-gnu-gcc"
    )
else:
    exec_module.export_library(output_file)

    # Validate on x86
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec_module, dev)

    input_data = np.random.rand(1, 3, 640, 640).astype("float32")
    tvm_input = tvm.runtime.tensor(input_data, dev)
    tvm_out = vm["main"](tvm_input)
    tvm_result = tvm_out[0].numpy()

    # Compare with PyTorch
    with torch.no_grad():
        pt_out = torch_model(torch.from_numpy(input_data))
        pt_result = pt_out[0].numpy()

    np.testing.assert_allclose(tvm_result, pt_result, rtol = 1e-4, atol = 1e-4)
    print("TVM output shape:", tvm_result.shape)
    print("PyTorch vs TVM outputs match!")

print(f"Compiled: {output_file}")