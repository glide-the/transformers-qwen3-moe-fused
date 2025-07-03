import torch
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


def silu_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return F.silu(x) * y
    return LigerSiLUMulFunction.apply(x, y)
