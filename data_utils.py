"""Utility helpers for dataset preprocessing and slice annotations."""

from __future__ import annotations

from typing import Dict, List
from unsloth.chat_templates import standardize_sharegpt


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
    """Normalise heterogeneous samples into standardize_sharegpt format."""

    slice_type = example.get("slice")
    conversations: List[Dict] = []

    if slice_type == "classification":
        text = example.get("text", "")
        label = str(example.get("label", ""))
        # 转换成 ShareGPT 标准格式
        conversations = [
            {"from": "system", "value": "分类任务: 请判断以下评论的情感。"},
            {"from": "user",   "value": f"评论: {text}"},
            {"from": "assistant", "value": f"答案: {label}"}
        ]

    elif slice_type == "agent":
        # agent 类型一般已经有 messages
        # 这里把 role/role-name 转换成 ShareGPT 格式
        raw_messages = example.get("messages", [])
        system_msg = example.get("system", "")

        if system_msg:
            conversations.append({"from": "system", "value": system_msg})

        for m in raw_messages:
            role = m.get("role")
            content = m.get("content", "")
            if role in ["user", "assistant", "system"]:
                conversations.append({"from": role, "value": content})
            else:
                # 默认 fallback 为 user
                conversations.append({"from": "user", "value": content})

    else:
        # fallback 情况
        text = example.get("text", "")
        conversations = [
            {"from": "user", "value": text}
        ]

    return {"conversations": conversations, "slice": slice_type}


from datasets import Dataset

def inspect_dataset(ds: Dataset, n: int = 5):
    """打印 HuggingFace Dataset 的状态面板"""
    print("="*60)
    print("📊 Dataset Info Panel")
    print("="*60)
    print(f"🔹 Num rows : {len(ds)}")
    print(f"🔹 Columns  : {ds.column_names}")
    print(f"🔹 Features : {ds.features}")
    print("="*60)
    print(f"🔹 Preview (前 {n} 条)")
    print("="*60)
    
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        print(f"Row {i}:")
        for col in ds.column_names:
            val = ex[col]
            # 截断太长的文本，避免刷屏
            if isinstance(val, str) and len(val) > 100:
                val = val[:100] + "..."
            print(f"  - {col}: {val}")
        print("-"*60)
