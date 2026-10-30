"""Cross-compile YOLOv8n for RISC-V (BPI-F3).

--conv2d llvm : conv2d compiled by LLVM (baseline)
--conv2d byoc : conv2d offloaded to the RVV BYOC runtime (kernel loaded at run time)

The rest of the model is always compiled by LLVM without +v, so the only
difference between the two outputs is the conv2d implementation.
"""
import argparse
from ultralytics import YOLO

import torch
from torch.export import export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions, RunCodegen
from byoc.patterns import get_conv2d_patterns

parser = argparse.ArgumentParser(description="Cross-compile YOLOv8n for RISC-V")
parser.add_argument("--conv2d", choices=["llvm", "byoc"], default="byoc",
                    help="conv2d implementation")
args = parser.parse_args()

# Import from PyTorch
torch_model = YOLO("yolov8n.pt").model.eval()
example_args = (torch.randn(1, 3, 640, 640),)

with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod = from_exported_program(
        exported_program,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=True,
    )

# No +v: the only vectorized code is the hand-written conv2d kernel.
target = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "riscv64-linux-gnu",
    "mcpu": "generic-rv64",
    "mabi": "lp64d",
    "mattr": ["+64bit", "+m", "+a", "+f", "+d", "+c"],
})
output_file = f"yolov8n_riscv_{args.conv2d}.so"

# Compile
passes = [
    relax.transform.DecomposeOpsForInference(),
    relax.transform.FoldBatchnormToConv2D(),
]
if args.conv2d == "byoc":
    passes += [
        FuseOpsByPattern(get_conv2d_patterns(), bind_constants=False, annotate_codegen=True),
        MergeCompositeFunctions(),
        RunCodegen(),
    ]
passes += [
    relax.transform.LegalizeOps(),
    relax.transform.AnnotateTIROpPattern(),
    relax.transform.FuseOps(),
    relax.transform.FuseTIR(),
]
mod_compiled = tvm.ir.transform.Sequential(passes)(mod)
exec_module = tvm.compile(mod_compiled, target=target)

exec_module.export_library(output_file, cc="riscv64-linux-gnu-gcc")
print(f"Compiled: {output_file}")