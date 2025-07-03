# See qwen3_moe_fused/functional.py for docstring
# Compared to moe_fused_linear_torch, this avoids allocating an intermediate array of shape (B, O, I)
# This can be improved, see:
# https://github.com/triton-lang/triton/blob/dd1c3d429d1c24904722ac699ea5750bc694c4d6/python/triton_kernels/triton_kernels/matmul_ogs.py
# https://github.com/ggml-org/llama.cpp/blob/a0535ffa0d35fccfec3e1a0a3bfc9dbb6054d7c0/ggml/src/ggml-cuda/ggml-cuda.cu#L2065
# https://github.com/vllm-project/vllm/blob/015fab8c2fa4db8776f7e91abd50371911673d88/vllm/model_executor/layers/fused_moe/fused_moe.py

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_I": 64, "BLOCK_SIZE_O": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 64}, num_warps=8, num_stages=3),
    ],
    key=["I", "O"],
)
@triton.jit
def dot_kernel(
    # Pointers
    x_ptr,
    w_ptr,
    s_ptr,
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
    stride_sb,
    stride_ob,
    stride_oo,
    # Metadata
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    b = tl.program_id(axis=0)
    e = tl.load(s_ptr + stride_sb * b)
    blk_idx_o = tl.program_id(axis=1)

    offs_o = (blk_idx_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)) % O
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    w_ptrs = w_ptr + stride_we * e + stride_wo * offs_o[:, None] + stride_wi * offs_i[None, :]
    x_ptrs = x_ptr + stride_xb * b + stride_xi * offs_i

    accumulator = tl.zeros((BLOCK_SIZE_O,), dtype=tl.float32)
    for blk_idx_i in range(tl.cdiv(I, BLOCK_SIZE_I)):
        mask_i = blk_idx_i * BLOCK_SIZE_I + offs_i < I
        _w = tl.load(w_ptrs, mask=mask_i[None, :], other=0.0)
        _x = tl.load(x_ptrs, mask=mask_i, other=0.0)

        # matrix-vector mul
        accumulator += tl.sum(_w * _x[None, :], axis=1)

        w_ptrs += stride_wi * BLOCK_SIZE_I
        x_ptrs += stride_xi * BLOCK_SIZE_I
    accumulator = accumulator.to(out_ptr.type.element_ty)

    offs_o = blk_idx_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    out_ptrs = out_ptr + stride_ob * b + stride_oo * offs_o
    mask_o = offs_o < O
    tl.store(out_ptrs, accumulator, mask=mask_o)


def index_matmul(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert w.device == x.device
    assert s.device == x.device
    assert w.dtype == x.dtype
    assert s.dtype == torch.int32
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

    out = torch.empty((B, O), dtype=x.dtype, device=x.device)
    grid = lambda META: (B, triton.cdiv(O, META["BLOCK_SIZE_O"]))
    dot_kernel[grid](
        # Pointers
        x,
        w,
        s,
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
        s.stride(0),
        out.stride(0),
        out.stride(1),
    )
    return out
