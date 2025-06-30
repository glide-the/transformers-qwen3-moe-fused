#!/usr/bin/env python3

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3MoeConfig
from trl import SFTConfig, SFTTrainer

from qwen3_moe_fused.lora import LoraMoeFusedLinear
from qwen3_moe_fused.modular_qwen3_moe_fused import (
    MoeFusedLinear,
    Qwen3MoeFusedForCausalLM,
)
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


def main():
    patch_bnb_quantizer()

    model_dir = "./pretrained/qwen-moe-tiny-lm"

    # Create a new model
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
    model = Qwen3MoeFusedForCausalLM(config)
    model.save_pretrained(model_dir)

    # Load and quantize the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3MoeFusedForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config)

    lora_config = LoraConfig(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # We can set a smaller rank for MoE layers
        # With rslora, we don't need to set a different alpha for them
        rank_pattern={
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
            "gate": 16,
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        },
        lora_alpha=1,
        use_rslora=True,
    )
    lora_config._register_custom_module({MoeFusedLinear: LoraMoeFusedLinear})
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    dataset = Dataset.from_dict(
        {
            "text": [
                "a" * 100,
                "b" * 100,
                "c" * 100,
                "d" * 100,
            ]
        }
    )

    sft_config = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-2,
        weight_decay=1e-2,
        num_train_epochs=1,
        logging_steps=1,
        bf16=True,
        optim="adamw_8bit",
        dataset_num_proc=1,
        max_length=1024,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer_stats = trainer.train()
    print("trainer_stats")
    print(trainer_stats)


if __name__ == "__main__":
    main()
