#!/usr/bin/env python3
#
# Example to train a LoRA on the fused and quantized version of Qwen3-30B-A3B using Unsloth

import os

from unsloth import FastModel

# Import unsloth before others
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    patch_bnb_quantizer()
    # We can set a smaller rank for MoE layers
    # With rslora, we don't need to set a different alpha for them
    # TODO: Support rank_pattern in Unsloth
    patch_lora_config(
        rank_pattern={
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
            # "gate": 16,  # It's possible to create a LoRA on the routing gate, but this is unstable
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        }
    )
    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    model_id = "bash99/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit"

    model, tokenizer = FastModel.from_pretrained(model_id, auto_model=Qwen3MoeFusedForCausalLM)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    dataset = load_dataset("stanfordnlp/imdb", split="train")
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, max_length=512),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./outputs/moe_unsloth",
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
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer_stats = trainer.train()
    print("trainer_stats")
    print(trainer_stats)


if __name__ == "__main__":
    main()
