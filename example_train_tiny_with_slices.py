#!/usr/bin/env python3
"""Example script demonstrating slice-aware fine-tuning for Qwen3-MoE."""

from __future__ import annotations

import os
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, TrainingArguments

from collators import SliceCollator
from curriculum import CurriculumCallback, CurriculumSampler
from data_utils import slice_by_metadata
from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer
from slice_trainer import SliceTrainer


os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")


def build_tiny_dataset() -> DatasetDict:
    samples = [
        {"text": "Hello world! This is a tiny English sentence."},
        {"text": "def add(a, b):\n    return a + b"},
        {"text": "你好，世界。这是一段中文样本。"},
        {"text": "class Greeter:\n    def greet(self):\n        print('hi')"},
        {"text": "纯中文示例，用于验证切片策略。"},
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "import math\nmath.sqrt(2)"},
        {"text": "Mixed language 示例 with ASCII and 非ASCII."},
        {"text": "Once upon a time in a tiny dataset."},
        {"text": "这是一个非常简短的中文描述。"},
    ]
    dataset = Dataset.from_list(samples)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict(
        train=dataset["train"].map(slice_by_metadata),
        validation=dataset["test"].map(slice_by_metadata),
    )
    return dataset


def main():
    patch_bnb_quantizer()
    patch_lora_config()
    patch_Qwen3MoeFusedSparseMoeBlock_forward()
    model_dir = "/media/checkpoint1/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"


    model = Qwen3MoeFusedForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    model = get_peft_model(model, lora_config)

    dataset = build_tiny_dataset()
 

    phases = [
        (1000, {"code": 0.6, "zh": 0.2, "en": 0.2}),
        (5000, {"code": 0.4, "zh": 0.3, "en": 0.3}),
        (10000, {"code": 0.3, "zh": 0.3, "en": 0.4}),
    ]
    curriculum_sampler = CurriculumSampler(dataset["train"], phases)
    curriculum_callback = CurriculumCallback(curriculum_sampler)

    data_collator = SliceCollator(tokenizer, max_seq_len=256, micro_batch_size=2)

    training_args = TrainingArguments(
        output_dir="./outputs/tiny_with_slices",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-3,
        weight_decay=1e-2,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=5,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SliceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[curriculum_callback],
        curriculum_sampler=curriculum_sampler,
    )

    trainer_stats = trainer.train()
    print("trainer_stats")
    print(trainer_stats)


if __name__ == "__main__":
    main()
