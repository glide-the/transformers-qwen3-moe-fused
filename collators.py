"""Custom data collators that are aware of slice metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class SliceCollator:
    """Collate samples grouped by their slice metadata.

    The collator keeps together samples that share the same ``slice`` tag so
    that each micro-batch emphasises a particular data regime. This makes it
    easier for the MoE router to specialise experts early in training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        capacity_factor: float = 1.2,
        max_seq_len: int = 512,
        micro_batch_size: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.capacity_factor = capacity_factor
        self.max_seq_len = max_seq_len
        self.micro_batch_size = micro_batch_size

    def _bucket_by_slice(self, examples: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for example in examples:
            slice_id = example.get("slice", "default")
            buckets[slice_id].append(example)
        return buckets

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        slice_buckets = self._bucket_by_slice(examples)

        input_ids: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []
        slice_ids: List[str] = []

        for slice_id, bucket in slice_buckets.items():
            texts = [example["text"] for example in bucket]
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            input_ids.extend(encodings["input_ids"])
            attention_masks.extend(encodings["attention_mask"])
            slice_ids.extend([slice_id] * len(texts))

        if not input_ids:
            raise ValueError("SliceCollator received an empty batch")

        input_ids_tensor = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask_tensor = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "slice_ids": slice_ids,
            "slice_count": Counter(slice_ids),
        }
