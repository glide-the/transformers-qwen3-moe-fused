"""Utility helpers for dataset preprocessing and slice annotations."""

from __future__ import annotations

from typing import Dict


def slice_by_metadata(example: Dict[str, str]) -> Dict[str, str]:
    """Annotate an example with a coarse language slice label.

    The heuristic matches the behaviour described in the task instructions and
    is intentionally lightweight so it can run during dataset mapping without
    additional dependencies.
    """

    text = example.get("text", "")
    if "def " in text or "class " in text:
        example["slice"] = "code"
    elif any(ch in text for ch in "。！？"):
        example["slice"] = "zh"
    elif text.isascii():
        example["slice"] = "en"
    else:
        example["slice"] = "general"
    return example
