#!/usr/bin/env python3
#
# Run test_model.py first

import torch

from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedModel
from transformers import BitsAndBytesConfig, Qwen3MoeModel, set_seed


def get_rtol_atol(actual, expect):
    diff = (actual - expect).abs()
    atol = diff.max()
    eps = torch.tensor(torch.finfo(actual.dtype).eps, device=actual.device, dtype=actual.dtype)
    rdiff = diff / torch.maximum(torch.maximum(actual.abs(), expect.abs()), eps)
    rtol = rdiff.max()
    return rtol, atol


def main():
    model_dir = "./pretrained/qwen-moe-tiny"
    model_quantized_dir = "./pretrained/qwen-moe-tiny-quantized"
    model_fused_dir = "./pretrained/qwen-moe-tiny-fused"
    model_fused_quantized_dir = "./pretrained/qwen-moe-tiny-fused-quantized"
    device = "cuda"
    set_seed(42)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model_quantized = Qwen3MoeModel.from_pretrained(model_dir, quantization_config=bnb_config)
    model_quantized.save_pretrained(model_quantized_dir)

    model_fused_quantized = Qwen3MoeFusedModel.from_pretrained(model_fused_dir, quantization_config=bnb_config)
    model_fused_quantized.save_pretrained(model_fused_quantized_dir)

    model = Qwen3MoeModel.from_pretrained(model_dir).to(device)
    model_quantized = Qwen3MoeModel.from_pretrained(model_quantized_dir).to(device)
    model_fused_quantized = Qwen3MoeFusedModel.from_pretrained(model_fused_quantized_dir).to(device)

    input_ids = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int32)
    hidden = model(input_ids=input_ids).last_hidden_state
    hidden_quantized = model_quantized(input_ids=input_ids).last_hidden_state
    hidden_fused_quantized = model_fused_quantized(input_ids=input_ids).last_hidden_state
    print(hidden)
    print(hidden_quantized)
    print(hidden_fused_quantized)
    print(get_rtol_atol(hidden_quantized, hidden))
    print(get_rtol_atol(hidden_fused_quantized, hidden))
    print(get_rtol_atol(hidden_fused_quantized, hidden_quantized))


if __name__ == "__main__":
    main()
