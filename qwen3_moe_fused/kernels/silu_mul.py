import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


def silu_mul(x, y):
    if not x.is_cuda:
        return F.silu(x) * y
    return LigerSiLUMulFunction.apply(x, y)
