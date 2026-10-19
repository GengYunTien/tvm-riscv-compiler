import tvm
from tvm import relax


def apply_conv2d_schedule(mod):
    """
    Apply TensorIR schedule to all conv2d TIR functions (NHWC layout).

    Strategy:
    - ConvertLayout: NCHW → NHWC, making ff (output channel) memory-contiguous
    - Split ff by TILE_F=16 (2× RVV vector registers, m2 mode: 2 × 8 float32)
    - Reorder: move reduction axes (ry, rx, rc) outside ff_inner
    - Vectorize ff_inner to hint LLVM to generate RVV vle32.v / vfmacc.vv

    Result: scalar -66% (42626ms → 14320ms), but RVV slower than scalar
    due to LLVM selecting m2 with vsetvli overhead vs scalar loop unrolling.
    → Hand-written RVV intrinsic (BYOC) required for true vectorization.
    """
    TILE_F = 16  # 2× RVV vector registers (m2): 2 × 8 float32 = 16 elements

    success = 0
    skip = 0

    for gv, func in mod.functions.items():
        if 'conv2d' in gv.name_hint:
            ff_size = int(func.params[-1].shape[1])
            if ff_size < TILE_F:
                skip += 1
                continue
            try:
                sch = tvm.s_tir.Schedule(func)
                block = sch.get_sblock('conv2d_nhwc')
                nn, yy, xx, ff, ry, rx, rc = sch.get_loops(block)
                ff_outer, ff_inner = sch.split(ff, factors=[None, TILE_F])
                sch.reorder(nn, yy, xx, ff_outer, ry, rx, rc, ff_inner)
                sch.vectorize(ff_inner)
                mod.update_func(gv, sch.mod['main'])
                success += 1
            except Exception as e:
                print(f"Warning: schedule failed for {gv.name_hint}: {e}")
                skip += 1

    print(f"Conv2d schedule: {success} scheduled, {skip} skipped")
    return mod