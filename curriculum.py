"""Curriculum sampling utilities to gradually adjust slice ratios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


@dataclass
class CurriculumPhase:
    """A single phase in the curriculum schedule."""

    threshold: int
    weights: Dict[str, float]


class CurriculumSampler:
    """Maintain slice sampling weights that evolve during training."""

    def __init__(
        self,
        dataset,
        phases: Sequence[Tuple[int, Dict[str, float]]],
        *,
        current_step: int = 0,
        default_weight: float = 1.0,
    ) -> None:
        if not phases:
            raise ValueError("CurriculumSampler requires at least one phase")
        self.dataset = dataset
        self.phases: List[CurriculumPhase] = [CurriculumPhase(*phase) for phase in phases]
        self.phases.sort(key=lambda phase: phase.threshold)
        self.current_step = int(current_step)
        self.default_weight = float(default_weight)

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def _current_phase_weights(self) -> Dict[str, float]:
        for phase in self.phases:
            if self.current_step <= phase.threshold:
                return phase.weights
        return self.phases[-1].weights

    def get_weights(self) -> torch.Tensor:
        num_examples = len(self.dataset)
        if num_examples == 0:
            return torch.zeros(0, dtype=torch.double)

        weights_map = self._current_phase_weights()
        if hasattr(self.dataset, "column_names") and "slice" in self.dataset.column_names:
            slices: Iterable[str] = self.dataset["slice"]
        else:
            slices = ["default"] * num_examples

        weights = [float(weights_map.get(slice_id, self.default_weight)) for slice_id in slices]
        tensor = torch.tensor(weights, dtype=torch.double)
        tensor = tensor.clamp_min(1e-6)
        return tensor


class CurriculumCallback(TrainerCallback):
    """Update the dataloader sampler as the curriculum progresses."""

    def __init__(self, sampler: CurriculumSampler) -> None:
        self.sampler = sampler

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.sampler.set_step(state.global_step)
        train_dataloader = kwargs.get("train_dataloader")
        if train_dataloader is None:
            return control

        data_sampler = getattr(train_dataloader, "sampler", None)
        if data_sampler is None or not hasattr(data_sampler, "weights"):
            return control

        weights = self.sampler.get_weights()
        if weights.numel() == 0:
            return control

        sampler_weights = data_sampler.weights
        dtype = sampler_weights.dtype if torch.is_tensor(sampler_weights) else torch.double
        device = sampler_weights.device if torch.is_tensor(sampler_weights) else torch.device("cpu")
        data_sampler.weights = weights.to(device=device, dtype=dtype)
        if hasattr(data_sampler, "num_samples"):
            data_sampler.num_samples = len(weights)

        return control
