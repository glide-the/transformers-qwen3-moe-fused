#!/usr/bin/env python3
#
# Example to train a tiny model using Unsloth
# Run example_create_tiny.py first
#
# If it shows `NameError: name 'Any' is not defined.`
# then we need to add a line:
# source = source.replace(": Any", "")
# after https://github.com/unslothai/unsloth-zoo/blob/362fb45ee5906052bf09a43f1052c578159069ac/unsloth_zoo/compiler.py#L1283
# See https://github.com/unslothai/unsloth/issues/2874

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
    # We can set a smaller rank for MoE layers
    # With rslora, we don't need to set a different alpha for them
    # TODO: Support rank_pattern in Unsloth
    patch_lora_config(
        rank_pattern={
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
            "gate": 16,
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        }
    )

    model_dir = "./pretrained/qwen-moe-tiny-lm-quantized"

    model, tokenizer = FastModel.from_pretrained(model_dir, auto_model=Qwen3MoeFusedForCausalLM)

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
        save_steps=3,
        bf16=True,
        optim="adamw_8bit",
        dataset_text_field="text",
        dataset_num_proc=1,
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
