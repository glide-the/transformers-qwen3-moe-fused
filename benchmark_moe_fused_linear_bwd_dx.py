#!/usr/bin/env python3

import gc
import os
from functools import partial
from math import sqrt

import torch
import triton

from qwen3_moe_fused.grouped_gemm.interface import grouped_gemm_dX
from qwen3_moe_fused.grouped_gemm.kernels_masked.forward_transposed import (
    grouped_gemm_forward_transposed,
)
from qwen3_moe_fused.kernels.indexing import get_expert_counts


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

providers = {
    "grouped_gemm": partial(grouped_gemm_dX, autotune=True),
    "grouped_gemm_masked": grouped_gemm_forward_transposed,
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
            plot_name="moe_fused_linear_bwd_dx",
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

    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (N,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)
    m_sizes = get_expert_counts(selected_experts, num_experts)
    grad_output = torch.randn(N, out_features, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: providers[provider](grad_output, weight, m_sizes), quantiles=quantiles
    )

    del weight, selected_experts, m_sizes, grad_output
    gc.collect()
    torch.cuda.empty_cache()

    perf = lambda ms: N * out_features * in_features / ms * 1e-6
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
