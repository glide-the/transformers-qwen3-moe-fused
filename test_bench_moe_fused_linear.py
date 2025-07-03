#!/usr/bin/env python3

import gc
from math import sqrt

import torch
import triton

from qwen3_moe_fused.functional import (
    moe_fused_linear_naive,
    moe_fused_linear_torch,
    moe_fused_linear_triton,
)


moe_fused_linear_torch_compiled = torch.compile(moe_fused_linear_torch, fullgraph=True, mode="max-autotune")


providers = {
    "naive": moe_fused_linear_naive,
    # "torch": moe_fused_linear_torch,  # This takes too much memory
    "compile": moe_fused_linear_torch_compiled,  # After compiling, this does not take much memory
    "triton": moe_fused_linear_triton,
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
            ylabel="GB/s",
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
    arg_bytes = sum([x.numel() + x.element_size() for x in [input, weight, selected_experts]])

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: providers[provider](input, weight, selected_experts), quantiles=quantiles
    )

    del input, weight, selected_experts
    gc.collect()
    torch.cuda.empty_cache()

    gbps = lambda ms: arg_bytes / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run(print_data=True, save_path="./")
