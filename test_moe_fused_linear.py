#!/usr/bin/env python3

import os
from math import sqrt

import torch

from qwen3_moe_fused.functional import _moe_fused_linear_naive_fwd, moe_fused_linear


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    batch_size = 1024
    in_features = 2048
    out_features = 768
    num_experts = 128
    device = "cuda"

    # For higher precision, use input_precision="ieee" in tl.dot in the kernels
    # dtype = torch.float32
    # rtol = 1e-6
    # atol = 1e-6

    # dtype = torch.float32
    # rtol = 1e-4
    # atol = 1e-4

    dtype = torch.bfloat16
    rtol = 1e-2
    atol = 1e-2

    input = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (batch_size,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)

    output_naive = _moe_fused_linear_naive_fwd(input, weight, selected_experts)
    print("output_naive", output_naive.shape, output_naive.dtype)

    output_grouped_gemm = moe_fused_linear(input, weight, selected_experts)
    print("output_grouped_gemm", output_grouped_gemm.shape, output_grouped_gemm.dtype)
    print(torch.allclose(output_grouped_gemm, output_naive, rtol=rtol, atol=atol))


if __name__ == "__main__":
    main()
