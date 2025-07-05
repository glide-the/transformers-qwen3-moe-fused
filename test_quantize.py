#!/usr/bin/env python3
#
# Run test_model.py first

import os

import torch
from transformers import BitsAndBytesConfig, Qwen3MoeModel, set_seed

from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedModel
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def get_rtol_atol(actual, expect):
    actual = actual.float()
    expect = expect.float()
    diff = (actual - expect).abs()
    eps = torch.tensor(torch.finfo(actual.dtype).eps, device=actual.device, dtype=actual.dtype)
    rdiff = diff / torch.maximum(torch.maximum(actual.abs(), expect.abs()), eps)
    return (
        f"mean_rtol={rdiff.mean().item():.3g} "
        f"max_rtol={rdiff.max().item():.3g} "
        f"mean_atol={diff.max().item():.3g} "
        f"max_atol={diff.max().item():.3g}"
    )


def main():
    patch_bnb_quantizer()

    model_dir = "./pretrained/qwen-moe-tiny"
    model_quantized_dir = "./pretrained/qwen-moe-tiny-quantized"
    model_fused_dir = "./pretrained/qwen-moe-tiny-fused"
    model_fused_quantized_dir = "./pretrained/qwen-moe-tiny-fused-quantized"
    device = "cuda"
    dtype = torch.bfloat16
    set_seed(42)

    vocab_size = 151936
    batch_size = 2
    seq_len = 64

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_quantized = Qwen3MoeModel.from_pretrained(
        model_dir, device_map=device, torch_dtype=dtype, quantization_config=bnb_config
    )
    model_quantized.save_pretrained(model_quantized_dir)

    model_fused_quantized = Qwen3MoeFusedModel.from_pretrained(
        model_fused_dir, device_map=device, torch_dtype=dtype, quantization_config=bnb_config
    )
    model_fused_quantized.save_pretrained(model_fused_quantized_dir)

    model = Qwen3MoeModel.from_pretrained(model_dir, device_map=device)
    model_quantized = Qwen3MoeModel.from_pretrained(model_quantized_dir, device_map=device)
    model_fused_quantized = Qwen3MoeFusedModel.from_pretrained(model_fused_quantized_dir, device_map=device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.int32)
    hidden = model(input_ids=input_ids).last_hidden_state
    hidden_quantized = model_quantized(input_ids=input_ids).last_hidden_state
    hidden_fused_quantized = model_fused_quantized(input_ids=input_ids).last_hidden_state
    # print(hidden)
    # print(hidden_quantized)
    # print(hidden_fused_quantized)
    print(get_rtol_atol(hidden_quantized, hidden))
    print(get_rtol_atol(hidden_fused_quantized, hidden))
    print(get_rtol_atol(hidden_fused_quantized, hidden_quantized))


if __name__ == "__main__":
    main()
