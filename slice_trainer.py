"""Custom Trainer implementations for slice-aware fine-tuning."""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer

from curriculum import CurriculumSampler


class SliceTrainer(Trainer):
    """Trainer that integrates curriculum-aware sampling and custom loss."""

    def __init__(
        self,
        *args: Any,
        curriculum_sampler: Optional[CurriculumSampler] = None,
        **kwargs: Any,
    ) -> None:
        self.curriculum_sampler = curriculum_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):   # ✅ 增加 train_dataset 参数
        if self.curriculum_sampler is None:
            return super()._get_train_sampler()

        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None

        weights = self.curriculum_sampler.get_weights()
        if weights.numel() == 0 or len(self.train_dataset) != len(weights):
            return super()._get_train_sampler()

        weights = weights.detach().cpu().to(dtype=torch.double)
        num_samples = len(weights)
        return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

    def compute_loss(self, 
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):

        slice_ids = inputs.pop("slice_ids", None)   # ✅ 取出来
        outputs = model(**inputs)
        from losses import compute_loss as moe_loss
        return moe_loss(model, {**inputs, "slice_ids": slice_ids}, return_outputs)
