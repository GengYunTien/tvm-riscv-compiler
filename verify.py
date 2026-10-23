import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Compare board output with x86 reference")
parser.add_argument("--ref", default="ref_out.bin")
parser.add_argument("--out", default="out.bin")
parser.add_argument("--shape", default="1,84,8400")
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--atol", type=float, default=1e-4)
args = parser.parse_args()

shape = tuple(int(s) for s in args.shape.split(","))
ref = np.fromfile(args.ref, dtype=np.float32)
out = np.fromfile(args.out, dtype=np.float32)

expected = int(np.prod(shape))
if ref.size != expected or out.size != expected:
    raise SystemExit(f"Size mismatch: ref={ref.size}, out={out.size}, expected={expected}")

diff = np.abs(out - ref)
print(f"max abs diff: {diff.max():.3e}, mean abs diff: {diff.mean():.3e}")
print(f"max rel diff: {(diff / (np.abs(ref) + 1e-12)).max():.3e}")

if np.allclose(out, ref, rtol=args.rtol, atol=args.atol):
    print("PASS: board output matches x86 reference")
else:
    bad = np.argwhere(~np.isclose(out, ref, rtol=args.rtol, atol=args.atol))
    print(f"FAIL: {len(bad)} / {expected} elements differ")
    raise SystemExit(1)