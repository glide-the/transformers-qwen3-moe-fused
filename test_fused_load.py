

from datasets import load_dataset
import os

import torch 
from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    model_fused_quantized_dir = "/media/checkpoint1/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"
    device = "cuda"

    vocab_size = 151936
    batch_size = 7
    seq_len = 13

    patch_bnb_quantizer()
    patch_lora_config()
    patch_Qwen3MoeFusedSparseMoeBlock_forward()
    model_fused_quantized = Qwen3MoeFusedForCausalLM.from_pretrained(model_fused_quantized_dir, device_map=device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.int32)
 
    hidden_fused_quantized = model_fused_quantized(input_ids=input_ids).last_hidden_state

    print(hidden_fused_quantized)

if __name__ == "__main__":
    main()
