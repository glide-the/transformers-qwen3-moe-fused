#!/usr/bin/env python3
"""Example training script for Qwen3-30B-A3B with Unsloth and slice-aware curriculum."""

from __future__ import annotations

import os

from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from unsloth import FastModel

from collators import SliceCollator
from curriculum import CurriculumCallback, CurriculumSampler
from data_utils import format_example, slice_by_metadata
from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer
from slice_trainer import SliceTrainer
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM


os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")


def main() -> None:
    # === Step 1. 打补丁 ===
    patch_bnb_quantizer()
    patch_lora_config()
    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    # === Step 2. 数据准备 ===
    imdb = load_from_disk(
        "/media/gpt4-pdf-chatbot-langchain/transformers-qwen3-moe-fused/dataset/imdb_train"
    )
    agent = load_dataset(
        "json",
        data_files="/media/gpt4-pdf-chatbot-langchain/transformers-qwen3-moe-fused/dataset/agent/gemini_q_glm_a_finetuning_events_1152q_1710191088.258371.json",
    )

    imdb = imdb.map(slice_by_metadata)
    agent = agent.map(slice_by_metadata)

    imdb = imdb.map(format_example)
    agent = agent.map(format_example)

    columns_to_keep = {"text", "slice"}
    imdb = imdb.remove_columns([col for col in imdb.column_names if col not in columns_to_keep])
    agent = agent.remove_columns([col for col in agent["train"].column_names if col not in columns_to_keep])

    train_dataset = concatenate_datasets([agent["train"], imdb])

    model_id = "/media/checkpoint1/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",  # 或 torch.float16
    )

    model, tokenizer = FastModel.from_pretrained(
        model_id,
        auto_model=Qwen3MoeFusedForCausalLM,
        quantization_config=quant_config,
        trust_remote_code=True,
    ) 

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = SliceCollator(tokenizer, max_seq_len=256, micro_batch_size=4)

    phases = [
        (1000, {"classification": 0.5, "agent": 0.5}),
        (5000, {"classification": 0.4, "agent": 0.6}),
        (10000, {"classification": 0.3, "agent": 0.7}),
    ]
    curriculum_sampler = CurriculumSampler(train_dataset, phases)
    curriculum_callback = CurriculumCallback(curriculum_sampler)

    try:
        model = FastModel.get_peft_model(
            model,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                # "gate",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            r=4,
            lora_alpha=1,
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        training_args = TrainingArguments(
            output_dir="./moe_unsloth",
            per_device_train_batch_size=1,  # Increase batch size if you have more memory
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            weight_decay=1e-3,  # For MoE models, weight decay can be smaller than dense models
            num_train_epochs=1,
            lr_scheduler_type="linear",
            warmup_steps=1000,
            logging_steps=1,
            save_steps=100,
            save_total_limit=5,
            bf16=True,
            optim="adamw_8bit",
            torch_compile=True,
            torch_compile_mode="max-autotune",
            report_to="none",  # You may report to Wandb
            seed=3407,

            remove_unused_columns=False,
        )

        trainer = SliceTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[curriculum_callback],
            curriculum_sampler=curriculum_sampler,
        )
        trainer.train()
    except Exception as e:
        import traceback

        print("❌ 训练过程中发生异常：", str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
