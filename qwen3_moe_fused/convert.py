#!/usr/bin/env python3

import json
import os
import re
from typing import Dict, Iterable, Optional, Union

import safetensors.torch
import torch
from huggingface_hub.serialization import save_torch_state_dict
from peft import LoraConfig
from tqdm import tqdm
from transformers import Qwen3MoeConfig


TStateDict = Dict[str, torch.Tensor]


def load_sharded_state_dict(save_directory: os.PathLike) -> TStateDict:
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


def convert_state_dict_to_fused_(
    state_dict: TStateDict, *, key_prefix: str, param_names: Iterable[str], num_hidden_layers: int, num_experts: int
) -> None:
    if not key_prefix and "layers.0.mlp.experts.0.down_proj.weight" not in state_dict.keys():
        key_prefix = "model."

    for layer_idx in range(num_hidden_layers):
        print(f"Layer {layer_idx}/{num_hidden_layers}")
        for param_name in param_names:
            params = []
            for expert_idx in tqdm(range(num_experts)):
                key = f"{key_prefix}layers.{layer_idx}.mlp.experts.{expert_idx}.{param_name}.weight"
                params.append(state_dict[key])
                del state_dict[key]
            key = f"{key_prefix}layers.{layer_idx}.mlp.{param_name}.weight"
            state_dict[key] = torch.stack(params)
            del params


def convert_state_dict_to_unfused_(
    state_dict: TStateDict, *, key_prefix: str, param_names: Iterable[str], num_hidden_layers: int, num_experts: int
) -> None:
    if not key_prefix and "layers.0.mlp.down_proj.weight" not in state_dict.keys():
        key_prefix = "model."

    for layer_idx in range(num_hidden_layers):
        print(f"Layer {layer_idx}/{num_hidden_layers}")
        for param_name in param_names:
            params_key = f"{key_prefix}layers.{layer_idx}.mlp.{param_name}.weight"
            params = state_dict[params_key]
            for expert_idx in tqdm(range(num_experts)):
                key = f"{key_prefix}layers.{layer_idx}.mlp.experts.{expert_idx}.{param_name}.weight"
                # A clone is needed, otherwise the state dict cannot be saved
                state_dict[key] = params[expert_idx].clone()
            del state_dict[params_key]
            del params


def convert_model_to_fused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    print(f"Loading {in_dir}")
    config = Qwen3MoeConfig.from_pretrained(in_dir)
    config.architectures[0] = config.architectures[0].replace("Qwen3Moe", "Qwen3MoeFused")
    state_dict = load_sharded_state_dict(in_dir)

    print("Converting...")
    convert_state_dict_to_fused_(
        state_dict,
        key_prefix="",
        param_names=["down_proj", "gate_proj", "up_proj"],
        num_hidden_layers=config.num_hidden_layers,
        num_experts=config.num_experts,
    )

    print(f"Saving {out_dir}")
    config.save_pretrained(out_dir)
    max_shard_size_kwarg = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    save_torch_state_dict(state_dict, out_dir, **max_shard_size_kwarg)


def convert_model_to_unfused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    print(f"Loading {in_dir}")
    config = Qwen3MoeConfig.from_pretrained(in_dir)
    config.architectures[0] = config.architectures[0].replace("Qwen3MoeFused", "Qwen3Moe")
    state_dict = load_sharded_state_dict(in_dir)

    print("Converting...")
    convert_state_dict_to_unfused_(
        state_dict,
        key_prefix="",
        param_names=["down_proj", "gate_proj", "up_proj"],
        num_hidden_layers=config.num_hidden_layers,
        num_experts=config.num_experts,
    )

    print(f"Saving {out_dir}")
    config.save_pretrained(out_dir)
    max_shard_size_kwarg = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    save_torch_state_dict(state_dict, out_dir, **max_shard_size_kwarg)


def convert_lora_to_fused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    print(f"Loading {in_dir}")
    config = LoraConfig.from_pretrained(in_dir)
    model_path = os.path.join(in_dir, "adapter_model.safetensors")
    state_dict = safetensors.torch.load_file(model_path)

    pattern = r"base_model\.model\.layers\.\d+\.mlp\.experts\.0\.down_proj\.lora_A\.weight"
    num_hidden_layers = len([x for x in state_dict.keys() if re.compile(pattern).fullmatch(x)])
    pattern = r"base_model\.model\.layers\.0\.mlp\.experts\.\d+\.down_proj\.lora_A\.weight"
    num_experts = len([x for x in state_dict.keys() if re.compile(pattern).fullmatch(x)])
    print("num_hidden_layers", num_hidden_layers, "num_experts", num_experts)

    print("Converting...")
    convert_state_dict_to_fused_(
        state_dict,
        key_prefix="base_model.model.",
        param_names=[
            "down_proj.lora_A",
            "down_proj.lora_B",
            "gate_proj.lora_A",
            "gate_proj.lora_B",
            "up_proj.lora_A",
            "up_proj.lora_B",
        ],
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
    )

    print(f"Saving {out_dir}")
    config.save_pretrained(out_dir)
    model_path = os.path.join(out_dir, "adapter_model.safetensors")
    safetensors.torch.save_file(state_dict, model_path)


def convert_lora_to_unfused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    print(f"Loading {in_dir}")
    config = LoraConfig.from_pretrained(in_dir)
    model_path = os.path.join(in_dir, "adapter_model.safetensors")
    state_dict = safetensors.torch.load_file(model_path)

    pattern = r"base_model\.model\.layers\.\d+\.mlp\.down_proj\.lora_A\.weight"
    num_hidden_layers = len([x for x in state_dict.keys() if re.compile(pattern).fullmatch(x)])
    num_experts = state_dict["base_model.model.layers.0.mlp.down_proj.lora_A.weight"].shape[0]
    print("num_hidden_layers", num_hidden_layers, "num_experts", num_experts)

    print("Converting...")
    convert_state_dict_to_unfused_(
        state_dict,
        key_prefix="base_model.model.",
        param_names=[
            "down_proj.lora_A",
            "down_proj.lora_B",
            "gate_proj.lora_A",
            "gate_proj.lora_B",
            "up_proj.lora_A",
            "up_proj.lora_B",
        ],
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
    )

    print(f"Saving {out_dir}")
    config.save_pretrained(out_dir)
    model_path = os.path.join(out_dir, "adapter_model.safetensors")
    safetensors.torch.save_file(state_dict, model_path)
