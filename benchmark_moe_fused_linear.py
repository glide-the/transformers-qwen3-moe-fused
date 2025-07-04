#!/usr/bin/env python3

import gc
import os
from math import sqrt

import torch
import triton

from qwen3_moe_fused.functional import (
    _moe_fused_linear_grouped_gemm_fwd,
    _moe_fused_linear_naive_fwd,
    _moe_fused_linear_torch_fwd_compiled,
    _moe_fused_linear_triton_batched_fwd,
    _moe_fused_linear_triton_fwd,
    _moe_fused_linear_triton_sorted_fwd,
)


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

providers = {
    "naive": _moe_fused_linear_naive_fwd,
    "compile": _moe_fused_linear_torch_fwd_compiled,
    "triton": _moe_fused_linear_triton_fwd,
    # "sorted": _moe_fused_linear_triton_sorted_fwd,
    # "batched": _moe_fused_linear_triton_batched_fwd,
    "grouped_gemm": _moe_fused_linear_grouped_gemm_fwd,
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
            plot_name="moe_fused_linear",
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
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (N,), device=device, dtype=torch.int32)
    # Assume selected_experts is sorted
    selected_experts, _ = torch.sort(selected_experts)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: providers[provider](input, weight, selected_experts), quantiles=quantiles
    )

    del input, weight, selected_experts
    gc.collect()
    torch.cuda.empty_cache()

    perf = lambda ms: N * out_features * in_features / ms * 1e-6
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
