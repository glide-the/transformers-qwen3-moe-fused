#!/usr/bin/env python3

from math import sqrt

import torch

from qwen3_moe_fused.functional import (
    _moe_fused_linear_naive_fwd,
    _moe_fused_linear_torch_fwd,
    _moe_fused_linear_triton_fwd,
)


def main():
    batch_size = 2
    in_features = 3
    out_features = 5
    num_experts = 7
    device = "cuda"
    dtype = torch.float32

    input = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (batch_size,), device=device, dtype=torch.int32)

    output_naive = _moe_fused_linear_naive_fwd(input, weight, selected_experts)
    print("output_naive", output_naive.shape, output_naive.dtype)

    output_torch = _moe_fused_linear_torch_fwd(input, weight, selected_experts)
    print("output_torch", output_torch.shape, output_torch.dtype)
    print(torch.allclose(output_torch, output_naive, rtol=1e-6, atol=1e-6))

    output_triton = _moe_fused_linear_triton_fwd(input, weight, selected_experts)
    print("output_triton", output_triton.shape, output_triton.dtype)
    print(torch.allclose(output_triton, output_naive, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    main()
