"""Generate a fixed input and the PyTorch reference output for board verification.
"""
import numpy as np
import torch
from ultralytics import YOLO

SEED = 0
SHAPE = (1, 3, 640, 640)

model = YOLO("yolov8n.pt").model.eval()

rng = np.random.default_rng(SEED)
x = rng.random(SHAPE, dtype=np.float32)
x.tofile("in.bin")

with torch.no_grad():
    y = model(torch.from_numpy(x))[0].numpy().astype(np.float32)
y.tofile("ref_out.bin")

print(f"in.bin: {x.shape} {x.dtype}")
print(f"ref_out.bin: {y.shape} {y.dtype}, range [{y.min():.3f}, {y.max():.3f}]")