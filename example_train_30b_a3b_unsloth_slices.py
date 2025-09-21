#!/usr/bin/env python3
"""Example training script for Qwen3-30B-A3B with Unsloth and slice-aware curriculum."""

from __future__ import annotations

import os
from typing import Optional

from datasets import load_from_disk
from datasets import load_dataset
from torch.utils.data import WeightedRandomSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel

from curriculum import CurriculumCallback, CurriculumSampler
from data_utils import format_example, slice_by_metadata, tokenize_fn
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
        if dataset is None:
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
    imdb = load_from_disk("/media/gpt4-pdf-chatbot-langchain/transformers-qwen3-moe-fused/dataset/imdb_train")
    agent = load_dataset("json", data_files="/media/gpt4-pdf-chatbot-langchain/transformers-qwen3-moe-fused/dataset/agent/gemini_q_glm_a_finetuning_events_1152q_1710191088.258371.json")

    imdb = imdb.map(slice_by_metadata)
    agent = agent.map(slice_by_metadata)

    imdb = imdb.map(format_example)
    agent = agent.map(format_example)

    columns_to_keep = {"prompt", "target", "slice"}
    imdb = imdb.remove_columns([col for col in imdb.column_names if col not in columns_to_keep])
    agent = agent.remove_columns([col for col in agent['train'].column_names if col not in columns_to_keep])

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets([agent['train'], imdb])
    # eval_dataset = imdb.get("test")

    model_id = "/media/checkpoint1/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = train_dataset.map(tokenize_fn, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    # if eval_dataset is not None:
    #     tokenized_eval = eval_dataset.map(tokenize_fn, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    # else:
    tokenized_eval = None

    data_collator_lm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def data_collator(features):
        filtered = [
            {key: value for key, value in feature.items() if key in {"input_ids", "attention_mask", "labels"}}
            for feature in features
        ]
        return data_collator_lm(filtered)

    phases = [
        (1000, {"classification": 0.5, "agent": 0.5}),
        (5000, {"classification": 0.4, "agent": 0.6}),
        (10000, {"classification": 0.3, "agent": 0.7}),
    ]
    curriculum_sampler = CurriculumSampler(tokenized_train, phases)
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        args=training_args,
        data_collator=data_collator,
        compute_loss_func=compute_loss,
        callbacks=[curriculum_callback],
        curriculum_sampler=curriculum_sampler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
