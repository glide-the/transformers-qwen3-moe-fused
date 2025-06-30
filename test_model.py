#!/usr/bin/env python3

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeModel, set_seed

from qwen3_moe_fused.convert import convert_model_to_fused, convert_model_to_unfused
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedModel


def main():
    model_dir = "./pretrained/qwen-moe-tiny"
    model_fused_dir = "./pretrained/qwen-moe-tiny-fused"
    model_roundtrip_dir = "./pretrained/qwen-moe-tiny-roundtrip"
    device = "cuda"
    dtype = torch.float32
    set_seed(42)
    max_shard_size = None

    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=5,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_window_layers=2,
        moe_intermediate_size=3,
        num_experts=9,
        norm_topk_prob=True,
    )

    model = Qwen3MoeModel(config).to(device, dtype)
    max_shard_size_kwarg = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    model.save_pretrained(model_dir, **max_shard_size_kwarg)

    convert_model_to_fused(model_dir, model_fused_dir, max_shard_size=max_shard_size)
    model_fused = Qwen3MoeFusedModel.from_pretrained(model_fused_dir, device_map=device, torch_dtype=dtype)

    convert_model_to_unfused(model_fused_dir, model_roundtrip_dir, max_shard_size=max_shard_size)
    model_roundtrip = Qwen3MoeModel.from_pretrained(model_roundtrip_dir, device_map=device, torch_dtype=dtype)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device, dtype=torch.int32)
    hidden = model(input_ids=input_ids).last_hidden_state
    hidden_fused = model_fused(input_ids=input_ids).last_hidden_state
    hidden_roundtrip = model_roundtrip(input_ids=input_ids).last_hidden_state
    # print(hidden.shape, hidden.device, hidden.dtype)
    # print(hidden_fused.shape, hidden_fused.device, hidden_fused.dtype)
    # print(hidden_roundtrip.shape, hidden_roundtrip.device, hidden_roundtrip.dtype)
    print(torch.allclose(hidden_fused, hidden, rtol=1e-7, atol=1e-7))
    print(torch.allclose(hidden_roundtrip, hidden, rtol=1e-7, atol=1e-7))


if __name__ == "__main__":
    main()
