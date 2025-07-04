#!/usr/bin/env python3

from math import sqrt

import torch

from qwen3_moe_fused.functional import (
    _moe_fused_linear_naive_bwd,
    _moe_fused_linear_torch_bwd,
    _moe_fused_linear_torch_fwd,
    _moe_fused_linear_triton_bwd,
)


def main():
    batch_size = 1024
    in_features = 3
    out_features = 5
    num_experts = 128
    device = "cuda"
    dtype = torch.float32

    input = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (batch_size,), device=device, dtype=torch.int32)
    selected_experts, _ = torch.sort(selected_experts)

    input_auto = input.clone().requires_grad_()
    weight_auto = weight.clone().requires_grad_()
    output_auto = _moe_fused_linear_torch_fwd(input_auto, weight_auto, selected_experts)

    grad_output = torch.randn(batch_size, out_features, device=device, dtype=dtype)
    output_auto.backward(gradient=grad_output)
    grad_input_auto = input_auto.grad
    grad_weight_auto = weight_auto.grad
    print("grad_input_auto", grad_input_auto.shape, grad_input_auto.dtype)
    print("grad_weight_auto", grad_weight_auto.shape, grad_weight_auto.dtype)

    grad_input_naive, grad_weight_naive, _ = _moe_fused_linear_naive_bwd(grad_output, input, weight, selected_experts)
    print("grad_input_naive", grad_input_naive.shape, grad_input_naive.dtype)
    print("grad_weight_naive", grad_weight_naive.shape, grad_weight_naive.dtype)
    print(torch.allclose(grad_input_naive, grad_input_auto, rtol=1e-6, atol=1e-6))
    print(torch.allclose(grad_weight_naive, grad_weight_auto, rtol=1e-6, atol=1e-6))

    grad_input_torch, grad_weight_torch, _ = _moe_fused_linear_torch_bwd(grad_output, input, weight, selected_experts)
    print("grad_input_torch", grad_input_torch.shape, grad_input_torch.dtype)
    print("grad_weight_torch", grad_weight_torch.shape, grad_weight_torch.dtype)
    print(torch.allclose(grad_input_torch, grad_input_auto, rtol=1e-6, atol=1e-6))
    print(torch.allclose(grad_weight_torch, grad_weight_auto, rtol=1e-6, atol=1e-6))

    grad_input_triton, grad_weight_triton, _ = _moe_fused_linear_triton_bwd(
        grad_output, input, weight, selected_experts
    )
    print("grad_input_triton", grad_input_triton.shape, grad_input_triton.dtype)
    print("grad_weight_triton", grad_weight_triton.shape, grad_weight_triton.dtype)
    print(torch.allclose(grad_input_triton, grad_input_auto, rtol=1e-6, atol=1e-6))
    print(torch.allclose(grad_weight_triton, grad_weight_auto, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    main()
