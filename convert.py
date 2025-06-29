#!/usr/bin/env python3

import json
import os
from typing import Dict, Optional, Union

import safetensors.torch
import torch
from huggingface_hub.serialization import save_torch_state_dict
from tqdm import tqdm

from qwen3_moe_fused.configuration_qwen3_moe_fused import Qwen3MoeFusedConfig
from transformers import Qwen3MoeConfig


def load_sharded_state_dict(save_directory: os.PathLike) -> Dict[str, torch.Tensor]:
    index_path = os.path.join(save_directory, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        model_path = os.path.join(save_directory, "model.safetensors")
        return safetensors.torch.load_file(model_path)

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    state_dict = {}
    for shard_file in tqdm(shard_files):
        # Load shard into memory
        shard_path = os.path.join(save_directory, shard_file)
        state_dict |= safetensors.torch.load_file(shard_path)
    return state_dict


def convert_model_to_fused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    config = Qwen3MoeConfig.from_pretrained(in_dir)
    config.architectures = ["Qwen3MoeFusedModel"]

    print(f"Loading {in_dir}")
    state_dict = load_sharded_state_dict(in_dir)

    print("Converting...")
    for layer_idx in range(config.num_hidden_layers):
        print(f"Layer {layer_idx}/{config.num_hidden_layers}")
        for param_name in ["down_proj", "gate_proj", "up_proj"]:
            params = []
            for expert_idx in tqdm(range(config.num_experts)):
                key = f"layers.{layer_idx}.mlp.experts.{expert_idx}.{param_name}.weight"
                params.append(state_dict[key])
                del state_dict[key]
            key = f"layers.{layer_idx}.mlp.{param_name}.weight"
            state_dict[key] = torch.stack(params)
            del params

    config.save_pretrained(out_dir)

    print(f"Saving {out_dir}")
    max_shard_size_kwarg = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    save_torch_state_dict(state_dict, out_dir, **max_shard_size_kwarg)


def convert_model_to_unfused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    config = Qwen3MoeFusedConfig.from_pretrained(in_dir)
    config.architectures = ["Qwen3MoeModel"]

    print(f"Loading {in_dir}")
    state_dict = load_sharded_state_dict(in_dir)

    print("Converting...")
    for layer_idx in range(config.num_hidden_layers):
        print(f"Layer {layer_idx}/{config.num_hidden_layers}")
        for param_name in ["down_proj", "gate_proj", "up_proj"]:
            params_key = f"layers.{layer_idx}.mlp.{param_name}.weight"
            params = state_dict[params_key]
            for expert_idx in tqdm(range(config.num_experts)):
                key = f"layers.{layer_idx}.mlp.experts.{expert_idx}.{param_name}.weight"
                # A clone is needed, otherwise the state dict cannot be saved
                state_dict[key] = params[expert_idx].clone()
            del state_dict[params_key]
            del params

    config.save_pretrained(out_dir)

    print(f"Saving {out_dir}")
    max_shard_size_kwarg = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    save_torch_state_dict(state_dict, out_dir, **max_shard_size_kwarg)
