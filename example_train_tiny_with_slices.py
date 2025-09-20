#!/usr/bin/env python3
"""Example script demonstrating slice-aware fine-tuning for Qwen3-MoE."""

from __future__ import annotations

import os
from typing import Optional

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from torch.utils.data import WeightedRandomSampler
from transformers import AutoTokenizer
from transformers.utils import has_length
from trl import SFTConfig, SFTTrainer

from collators import SliceCollator
from curriculum import CurriculumCallback, CurriculumSampler
from data_utils import slice_by_metadata
from losses import compute_loss
from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")


class SliceSFTTrainer(SFTTrainer):
    """SFTTrainer variant that injects curriculum-aware sampling."""

    def __init__(self, *args, curriculum_sampler: Optional[CurriculumSampler] = None, **kwargs):
        self.curriculum_sampler = curriculum_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if self.curriculum_sampler is None:
            return super()._get_train_sampler(train_dataset)

        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or not has_length(dataset):
            return None

        weights = self.curriculum_sampler.get_weights()
        if weights.numel() == 0:
            return super()._get_train_sampler(train_dataset)

        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


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

    model_dir = "./pretrained/qwen-moe-tiny-lm"

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

    collator = SliceCollator(tokenizer, max_seq_len=256, micro_batch_size=4)

    phases = [
        (1000, {"code": 0.6, "zh": 0.2, "en": 0.2}),
        (5000, {"code": 0.4, "zh": 0.3, "en": 0.3}),
        (10000, {"code": 0.3, "zh": 0.3, "en": 0.4}),
    ]
    curriculum_sampler = CurriculumSampler(dataset["train"], phases)
    curriculum_callback = CurriculumCallback(curriculum_sampler)

    sft_config = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-3,
        weight_decay=1e-2,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=5,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        dataset_text_field="text",
        dataset_num_proc=1,
        report_to="none",
    )

    trainer = SliceSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=sft_config,
        data_collator=collator,
        compute_loss_func=compute_loss,
        callbacks=[curriculum_callback],
        curriculum_sampler=curriculum_sampler,
    )

    trainer_stats = trainer.train()
    print("trainer_stats")
    print(trainer_stats)


if __name__ == "__main__":
    main()
