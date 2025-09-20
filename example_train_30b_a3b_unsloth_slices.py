#!/usr/bin/env python3
"""Example training script for Qwen3-30B-A3B with Unsloth and slice-aware curriculum."""

from __future__ import annotations

import os
from typing import Optional

from datasets import load_dataset
from torch.utils.data import WeightedRandomSampler
from transformers import TrainingArguments
from transformers.utils import has_length
from trl import SFTTrainer

from unsloth import FastLanguageModel

from collators import SliceCollator
from curriculum import CurriculumCallback, CurriculumSampler
from data_utils import slice_by_metadata
from losses import compute_loss
from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
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


def main() -> None:
    # === Step 1. 打补丁 ===
    patch_bnb_quantizer()
    patch_lora_config(
        rank_pattern={
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        }
    )
    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    # === Step 2. 数据准备 ===
    dataset = load_dataset("stanfordnlp/imdb")
    dataset = dataset.map(slice_by_metadata)

    model_id = "bash99/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"
    tokenizer = FastLanguageModel.get_tokenizer(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = SliceCollator(tokenizer)

    phases = [
        (1000, {"en": 0.5, "zh": 0.3, "code": 0.2}),
        (5000, {"en": 0.4, "zh": 0.3, "code": 0.3}),
        (10000, {"en": 0.3, "zh": 0.3, "code": 0.4}),
    ]
    curriculum_sampler = CurriculumSampler(dataset["train"], phases)
    curriculum_callback = CurriculumCallback(curriculum_sampler)

    # === Step 3. 加载模型 ===
    model = FastLanguageModel.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=16,
        lora_alpha=16,
        use_rslora=True,
        modules_to_save=None,
        rank_pattern={
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        },
        use_gradient_checkpointing="unsloth",
    )

    # === Step 4. Trainer ===
    training_args = TrainingArguments(
        output_dir="./moe_unsloth",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=1000,
        bf16=True,
        report_to="none",
    )

    trainer = SliceSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        args=training_args,
        data_collator=collator,
        compute_loss_func=compute_loss,
        callbacks=[curriculum_callback],
        curriculum_sampler=curriculum_sampler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
