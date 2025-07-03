import torch
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


def silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.is_cuda:
        return LigerSiLUMulFunction.apply(x, y)
    else:
        return F.silu(x) * y
