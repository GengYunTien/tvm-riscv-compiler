from tvm.relax import dpl


def get_conv2d_patterns():
    """
    Define Relax dataflow patterns for RVV conv2d BYOC backend.

    Returns a list of (pattern_name, pattern) tuples to be used with
    FuseOpsByPattern. Pattern names use the prefix "rvv." to identify
    this backend.
    """
    data = dpl.wildcard()
    weight = dpl.wildcard()
    conv2d = dpl.is_op("relax.nn.conv2d")(data, weight)

    return [("rvv.conv2d", conv2d)]