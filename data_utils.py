"""Utility helpers for dataset preprocessing and slice annotations."""

from __future__ import annotations

from typing import Dict, List


def slice_by_metadata(example: Dict) -> Dict:
    """Assign a slice label based on the structure of the example."""

    if "messages" in example:
        example["slice"] = "agent"
    elif "text" in example and "label" in example:
        example["slice"] = "classification"
    else:
        example["slice"] = "general"
    return example


def format_example(example: Dict) -> Dict:
    """Normalise heterogeneous samples into prompt/target pairs."""

    if example.get("slice") == "classification":
        text = example["text"]
        label = str(example["label"])
        text = f"分类任务: 请判断以下评论的情感。\n\n评论: {text}\n\n答案: {label}".format(text=text, label=label)
       
    elif example.get("slice") == "agent":
        messages = example.get("messages", [])
        conv = ""
        for message in messages[:-1]:
            role = "用户" if message.get("role") == "user" else "助手"
            content = message.get("content", "")
            conv += f"{role}: {content}\n" if content else f"{role}: \n"
        system = example.get("system", "")
        text = f"{system}\n{conv}" if system else conv
    
    else:
        prompt = example.get("text", "")
        target = ""

    return {"text": text, "slice": example.get("slice")}


def tokenize_fn(examples: Dict[str, List[str]], tokenizer, max_length: int = 512):
    """Tokenize prompt/target pairs for supervised fine-tuning."""

    prompts: List[str] = examples["prompt"]
    targets: List[str] = examples["target"]
    texts = [prompt + target for prompt, target in zip(prompts, targets)]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings
