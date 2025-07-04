# out[b, o] = sum_i w[s[b], o, i] * x[b, i]
# Assume s is sorted, so for each expert, we only need to sum over a slice of b

from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import compute_batch_begins_ends


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_B": 256, "BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=2),
    ],
    key=["I", "O"],
)
@triton.jit
def _index_matmul_batched_kernel(
    # Pointers
    x_ptr,
    w_ptr,
    s_begins_ends_ptr,
    out_ptr,
    # Dimensions
    I,
    O,
    # Strides
    stride_xb,
    stride_xi,
    stride_we,
    stride_wo,
    stride_wi,
    stride_se,
    stride_s1,
    stride_ob,
    stride_oo,
    # Metadata
    BLOCK_SIZE_B: tl.constexpr = 256,
    BLOCK_SIZE_I: tl.constexpr = 64,
    BLOCK_SIZE_O: tl.constexpr = 64,
):
    e = tl.program_id(axis=0)
    b_begin = tl.load(s_begins_ends_ptr + stride_se * e)
    b_end = tl.load(s_begins_ends_ptr + stride_se * e + stride_s1)
    if b_begin >= b_end:
        return

    blk_idx_b = tl.program_id(axis=1)
    offs_b = blk_idx_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = (offs_b >= b_begin) & (offs_b < b_end)

    blk_idx_o = tl.program_id(axis=2)
    offs_o = (blk_idx_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) % O
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    w_ptrs = w_ptr + stride_we * e + stride_wo * offs_o[:, None] + stride_wi * offs_i[None, :]
    x_ptrs = x_ptr + stride_xb * offs_b[:, None] + stride_xi * offs_i[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    for blk_idx_i in range(tl.cdiv(I, BLOCK_SIZE_I)):
        mask_i = offs_i < I
        _w = tl.load(w_ptrs, mask=mask_i[None, :], other=0.0)
        _x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_i, other=0.0)

        accumulator = tl.dot(_x, _w.T, accumulator)
        # For testing with float32, use:
        # accumulator = tl.dot(_x, _w.T, accumulator, input_precision="ieee")

        offs_i += BLOCK_SIZE_I
        w_ptrs += stride_wi * BLOCK_SIZE_I
        x_ptrs += stride_xi * BLOCK_SIZE_I
    accumulator = accumulator.to(out_ptr.type.element_ty)

    offs_o = blk_idx_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    out_ptrs = out_ptr + stride_ob * offs_b[:, None] + stride_oo * offs_o[None, :]
    mask_o = offs_o < O
    tl.store(out_ptrs, accumulator, mask=mask_b[:, None] & mask_o[None, :])


def index_matmul_batched(
    x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    assert x.is_cuda
    assert w.device == x.device
    assert s.device == x.device
    assert s.dtype in [torch.int32, torch.int64]
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert s.is_contiguous()
    assert x.ndim == 2
    assert w.ndim == 3
    assert s.ndim == 1
    E, O, I = w.shape
    B = s.numel()
    assert x.shape[0] == B
    assert x.shape[1] == I

    # It's possible to reuse s_begins_ends in multiple MoeFusedLinear layers that use the same s,
    # but for now we recompute it for clarity
    s_begins_ends = compute_batch_begins_ends(s, E)

    if dtype is None:
        dtype = x.dtype
    out = torch.empty((B, O), device=x.device, dtype=dtype)
    grid = lambda META: (
        E,
        triton.cdiv(B, META["BLOCK_SIZE_B"]),
        triton.cdiv(O, META["BLOCK_SIZE_O"]),
    )
    _index_matmul_batched_kernel[grid](
        # Pointers
        x,
        w,
        s_begins_ends,
        out,
        # Dimensions
        I,
        O,
        # Strides
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        s_begins_ends.stride(0),
        s_begins_ends.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out
