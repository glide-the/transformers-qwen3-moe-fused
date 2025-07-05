#!/usr/bin/env python3

import os
from math import sqrt

import torch

from qwen3_moe_fused.functional import (
    _moe_fused_linear_naive_bwd_input,
    _moe_fused_linear_naive_bwd_weight,
    _moe_fused_linear_naive_fwd,
    moe_fused_linear,
)
from test_quantize import get_rtol_atol


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
    grad_output = torch.randn(batch_size, out_features, device=device, dtype=dtype)

    input_auto = input.clone().requires_grad_()
    weight_auto = weight.clone().requires_grad_()
    output_auto = _moe_fused_linear_naive_fwd(input_auto, weight_auto, selected_experts)
    output_auto.backward(gradient=grad_output)
    grad_input_auto = input_auto.grad
    grad_weight_auto = weight_auto.grad
    print("grad_input_auto", grad_input_auto.shape, grad_input_auto.dtype)
    print("grad_weight_auto", grad_weight_auto.shape, grad_weight_auto.dtype)

    grad_input_naive = _moe_fused_linear_naive_bwd_input(grad_output, input, weight, selected_experts)
    grad_weight_naive = _moe_fused_linear_naive_bwd_weight(grad_output, input, weight, selected_experts)
    print("grad_input_naive", grad_input_naive.shape, grad_input_naive.dtype)
    print("grad_weight_naive", grad_weight_naive.shape, grad_weight_naive.dtype)
    print(torch.allclose(grad_input_naive, grad_input_auto, rtol=rtol, atol=atol))
    print(torch.allclose(grad_weight_naive, grad_weight_auto, rtol=rtol, atol=atol))
    print(get_rtol_atol(grad_input_naive, grad_input_auto))
    print(get_rtol_atol(grad_weight_naive, grad_weight_auto))

    input_grouped_gemm = input.clone().requires_grad_()
    weight_grouped_gemm = weight.clone().requires_grad_()
    output_grouped_gemm = moe_fused_linear(input_grouped_gemm, weight_grouped_gemm, selected_experts)
    output_grouped_gemm.backward(gradient=grad_output)
    grad_input_grouped_gemm = input_grouped_gemm.grad
    grad_weight_grouped_gemm = weight_grouped_gemm.grad
    print("grad_input_grouped_gemm", grad_input_grouped_gemm.shape, grad_input_grouped_gemm.dtype)
    print("grad_weight_grouped_gemm", grad_weight_grouped_gemm.shape, grad_weight_grouped_gemm.dtype)
    print(torch.allclose(grad_input_grouped_gemm, grad_input_auto, rtol=rtol, atol=atol))
    print(torch.allclose(grad_weight_grouped_gemm, grad_weight_auto, rtol=rtol, atol=atol))
    print(get_rtol_atol(grad_input_grouped_gemm, grad_input_auto))
    print(get_rtol_atol(grad_weight_grouped_gemm, grad_weight_auto))


if __name__ == "__main__":
    main()
