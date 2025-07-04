# out[e, o, i] = sum_b if(s[b] == e) y[b, o] * x[b, i]
# Assume s is sorted, so for each expert, we only need to sum over a slice of b

from functools import partial

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_B": 32, "BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128}, num_warps=8, num_stages=3),
    ],
    key=["I", "O"],
)
@triton.jit
def _matmul_scatter_add_kernel(
    # Pointers
    x_ptr,
    y_ptr,
    s_begins_ends_ptr,
    out_ptr,
    # Dimensions
    I,
    O,
    # Strides
    stride_xb,
    stride_xi,
    stride_yb,
    stride_yo,
    stride_se,
    stride_s1,
    stride_oe,
    stride_oo,
    stride_oi,
    # Metadata
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
) -> None:
    e = tl.program_id(axis=0)
    b_begin = tl.load(s_begins_ends_ptr + stride_se * e)
    b_end = tl.load(s_begins_ends_ptr + stride_se * e + stride_s1)
    if b_begin >= b_end:
        return

    blk_idx_b = b_begin // BLOCK_SIZE_B
    blk_idx_b_end = tl.cdiv(b_end, BLOCK_SIZE_B)
    offs_b = blk_idx_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

    blk_idx_o = tl.program_id(axis=1)
    blk_idx_i = tl.program_id(axis=2)
    offs_o = (blk_idx_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) % O
    offs_i = (blk_idx_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)) % I
    x_ptrs = x_ptr + stride_xb * offs_b[:, None] + stride_xi * offs_i[None, :]
    y_ptrs = y_ptr + stride_yb * offs_b[:, None] + stride_yo * offs_o[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    while blk_idx_b < blk_idx_b_end:
        mask_b = (offs_b >= b_begin) & (offs_b < b_end)
        mask_b = mask_b[:, None]
        _x = tl.load(x_ptrs, mask=mask_b, other=0.0)
        _y = tl.load(y_ptrs, mask=mask_b, other=0.0)

        accumulator = tl.dot(_y.T, _x, accumulator)
        # For testing with float32, use:
        # accumulator = tl.dot(_y.T, _x, accumulator, input_precision="ieee")

        blk_idx_b += 1
        offs_b += BLOCK_SIZE_B
        x_ptrs += stride_xb * BLOCK_SIZE_B
        y_ptrs += stride_yb * BLOCK_SIZE_B
    accumulator = accumulator.to(out_ptr.type.element_ty)

    offs_o = blk_idx_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = blk_idx_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    out_ptrs = out_ptr + stride_oe * e + stride_oo * offs_o[:, None] + stride_oi * offs_i[None, :]
    mask_o = offs_o < O
    mask_i = offs_i < I
    tl.store(out_ptrs, accumulator, mask=mask_o[:, None] & mask_i[None, :])


@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _compute_batch_begins_ends(s: torch.Tensor, E: int) -> torch.Tensor:
    arange = torch.arange(E, device=s.device, dtype=s.dtype)
    s_begins = (arange[:, None] > s[None, :]).to(torch.int32).sum(dim=1)
    s_ends = (arange[:, None] >= s[None, :]).to(torch.int32).sum(dim=1)
    s_begins_ends = torch.stack([s_begins, s_ends], dim=1)
    return s_begins_ends


def matmul_scatter_add(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor, E: int, dtype: torch.dtype) -> torch.Tensor:
    assert x.is_cuda
    assert y.device == x.device
    assert s.device == x.device
    assert s.dtype in [torch.int32, torch.int64]
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert s.is_contiguous()
    assert x.ndim == 2
    assert y.ndim == 2
    assert s.ndim == 1
    B, I = x.shape
    _, O = y.shape
    assert y.shape[0] == B
    assert s.numel() == B

    # It's possible to reuse s_begins_ends in multiple MoeFusedLinear layers that use the same s,
    # but for now we recompute it for clarity
    s_begins_ends = _compute_batch_begins_ends(s, E)

    out = torch.zeros((E, O, I), device=x.device, dtype=dtype)
    grid = lambda META: (
        E,
        triton.cdiv(O, META["BLOCK_SIZE_O"]),
        triton.cdiv(I, META["BLOCK_SIZE_I"]),
    )
    _matmul_scatter_add_kernel[grid](
        # Pointers
        x,
        y,
        s_begins_ends,
        out,
        # Dimensions
        I,
        O,
        # Strides
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        s_begins_ends.stride(0),
        s_begins_ends.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out
