#!/usr/bin/env python3

import gc
import os
from functools import partial

import torch
import triton

from qwen3_moe_fused.gemv.matmul_scatter_add import matmul_scatter_add
from qwen3_moe_fused.grouped_gemm.interface import grouped_gemm_dW
from qwen3_moe_fused.grouped_gemm.kernels_masked.backward_dw import (
    grouped_gemm_backward_dw,
)
from qwen3_moe_fused.kernels.indexing import get_expert_counts


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

providers = {
    "gemv": partial(matmul_scatter_add, dtype=torch.bfloat16),
    "grouped_gemm": partial(grouped_gemm_dW, autotune=True),
    "grouped_gemm_masked": partial(grouped_gemm_backward_dw, dtype=torch.bfloat16),
}
provider_names = list(providers)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=range(1024, 16384 + 1, 1024),
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="GFLOPS",
            plot_name="moe_fused_linear_bwd_dw",
            args={},
        )
    ]
)
def benchmark(N, provider):
    print("N", N, "provider", provider)

    in_features = 2048
    out_features = 768
    num_experts = 128
    device = "cuda"
    dtype = torch.bfloat16

    input = torch.randn(N, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (N,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)
    m_sizes = get_expert_counts(selected_experts, num_experts)
    grad_output = torch.randn(N, out_features, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "gemv":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: providers[provider](input, grad_output, selected_experts, num_experts), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: providers[provider](input, grad_output, m_sizes), quantiles=quantiles
        )

    del input, selected_experts, m_sizes, grad_output
    gc.collect()
    torch.cuda.empty_cache()

    perf = lambda ms: N * out_features * in_features / ms * 1e-6
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
