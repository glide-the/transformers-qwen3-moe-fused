#!/usr/bin/env python3

import torch

from convert import convert_model_to_fused, convert_model_to_unfused
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedModel
from transformers import Qwen3MoeConfig, Qwen3MoeModel, set_seed


def main():
    model_dir = "./pretrained/qwen-moe-tiny"
    model_fused_dir = "./pretrained/qwen-moe-tiny-fused"
    model_roundtrip_dir = "./pretrained/qwen-moe-tiny-roundtrip"
    device = "cuda"
    dtype = torch.bfloat16
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
    model_fused = Qwen3MoeFusedModel.from_pretrained(model_fused_dir).to(device, dtype)

    convert_model_to_unfused(model_fused_dir, model_roundtrip_dir, max_shard_size=max_shard_size)
    model_roundtrip = Qwen3MoeModel.from_pretrained(model_roundtrip_dir).to(device, dtype)

    input_ids = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int32)
    hidden = model(input_ids=input_ids).last_hidden_state
    hidden_fused = model_fused(input_ids=input_ids).last_hidden_state
    hidden_roundtrip = model_roundtrip(input_ids=input_ids).last_hidden_state
    # print(hidden.shape, hidden.device, hidden.dtype)
    # print(hidden_fused.shape, hidden_fused.device, hidden_fused.dtype)
    # print(hidden_roundtrip.shape, hidden_roundtrip.device, hidden_roundtrip.dtype)
    print(torch.allclose(hidden_fused, hidden, rtol=1e-6, atol=1e-6))
    print(torch.allclose(hidden_roundtrip, hidden, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    main()
