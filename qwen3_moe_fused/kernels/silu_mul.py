# h = silu(e) * g
# Modified from https://github.com/unslothai/unsloth/blob/01c5e1a24935b93d9d3197815ad71751d9dfb37a/unsloth/kernels/swiglu.py

import torch
import triton
import triton.language as tl


_autotune_configs = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
]


@triton.autotune(
    configs=_autotune_configs,
    key=["n_elements"],
)
@triton.jit
def _fg_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)  # .to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row)  # e_row / (1 + tl.exp(-e_row))
    f_row = f_row.to(g_row.dtype)  # Exact copy from HF
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask=mask)


@triton.autotune(
    configs=_autotune_configs,
    key=["n_elements"],
)
@triton.jit
def _DWf_DW_dfg_kernel(
    DW,
    e,
    g,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))
    f = (se * e).to(dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask=mask, other=0)  # .to(tl.float32)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)  # .to(tl.float32)

    # e = e.float()
    # se = 1.0 / (1.0 + torch.exp(-e))
    se_row = tl.sigmoid(e_row)  # 1.0 / (1.0 + tl.exp(-e_row))
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    # h_row = f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row
    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives in buffers
    # tl.store(DW + offsets, h_row, mask=mask)  # h  = f * g
    tl.store(e + offsets, df_row, mask=mask)  # df = DW * f
    tl.store(g + offsets, de_row, mask=mask)  # de


class SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, g):
        ctx.save_for_backward(e, g)
        n_elements = e.numel()
        h = torch.empty_like(e)
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _fg_kernel[grid](e, g, h, n_elements)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        e, g = ctx.saved_tensors
        n_elements = e.numel()
        # e, g are modified in place to grad_e, grad_g
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _DWf_DW_dfg_kernel[grid](grad_h, e, g, n_elements)
        return e, g


silu_mul = SiluMul.apply
