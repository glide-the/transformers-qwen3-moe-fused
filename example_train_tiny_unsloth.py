#!/usr/bin/env python3
#
# Example to train a tiny model using Unsloth
# Run example_create_tiny.py first

import os

from unsloth import FastModel

# Import unsloth before others
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def main():
    patch_bnb_quantizer()
    patch_lora_config()

    model_dir = "./pretrained/qwen-moe-tiny-lm"

    model, tokenizer = FastModel.from_pretrained(model_dir, auto_model=Qwen3MoeFusedForCausalLM)

    # TODO: Support rank_pattern in Unsloth
    model = FastModel.get_peft_model(
        model,
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
        r=4,
        lora_alpha=1,
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = Dataset.from_dict({"text": [x * 100 for x in "abcdefghijkl"]})

    # These hyperparameters are for exaggerating the training of the tiny model
    # Don't use them in actual training
    sft_config = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-2,
        weight_decay=1e-2,
        num_train_epochs=1,
        logging_steps=1,
        bf16=True,
        optim="adamw_8bit",
        dataset_text_field="text",
        dataset_num_proc=1,
        max_length=1024,
        report_to="none",
        seed=3407,
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
