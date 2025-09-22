"""Auxiliary loss helpers for mixture-of-experts regularisation."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


def balance_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Encourage a uniform expert load distribution."""

    probs = F.softmax(router_logits, dim=-1)
    expert_load = probs.sum(0) / probs.sum()
    target = torch.full_like(expert_load, 1.0 / expert_load.size(0))
    return F.mse_loss(expert_load, target)


def contrastive_loss(router_logits: torch.Tensor, slice_ids: Sequence[str]) -> torch.Tensor:
    """Pull together router decisions for the same slice and push apart others."""

    probs = F.softmax(router_logits, dim=-1)

    # Router logits may carry extra structural dimensions (e.g. groups, top-k
    # experts).  We only care about a single representation per batch element, so
    # fold the remaining axes into the expert dimension.  Additionally ensure
    # there is always an explicit batch dimension to keep the pairwise
    # similarity computation stable.
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    elif probs.dim() > 2:
        probs = probs.reshape(probs.size(0), -1)


    sim = torch.matmul(probs, probs.T)
    loss = torch.zeros((), dtype=probs.dtype, device=probs.device)
    count = 0
    for i in range(len(slice_ids)):
        for j in range(i + 1, len(slice_ids)):
            if slice_ids[i] == slice_ids[j]:
                loss = loss + (1 - sim[i, j]) ** 2
            else:
                loss = loss + (sim[i, j]) ** 2
            count += 1
    if count == 0:
        return torch.zeros((), dtype=probs.dtype, device=probs.device)
    return loss / count


def _reshape_router_logits(
    router_logits: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    return router_logits.view(batch_size, seq_len, -1)


def _aggregate_router_logits(
    router_logits: Iterable[torch.Tensor],
    batch_size: int,
    seq_len: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    tensors: List[torch.Tensor] = []
    for layer_router in router_logits:
        if layer_router is None:
            continue
        tensors.append(_reshape_router_logits(layer_router, batch_size, seq_len))

    if not tensors:
        return None

    stacked = torch.stack(tensors, dim=0)  # (num_layers, batch, seq, experts)
    stacked = stacked.permute(1, 0, 2, 3)  # (batch, num_layers, seq, experts)

    if attention_mask is not None:
        mask = attention_mask.to(stacked.device).unsqueeze(1).unsqueeze(-1)
        token_totals = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        stacked = (stacked * mask).sum(dim=2) / token_totals
    else:
        stacked = stacked.mean(dim=2)

    return stacked.mean(dim=1)  # (batch, experts)


def compute_loss(model, inputs, return_outputs: bool = False):
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        labels=inputs["input_ids"],
        output_router_logits=True,
    )
    main_loss = outputs["loss"]
    router_logits = outputs.get("router_logits")
    slice_ids = inputs.get("slice_ids")
    attention_mask = inputs.get("attention_mask")

    aux_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
    if router_logits is not None and slice_ids is not None:
        batch_size, seq_len = inputs["input_ids"].shape[:2]
        if isinstance(router_logits, torch.Tensor):
            router_iterable = (router_logits,)
        else:
            router_iterable = tuple(router_logits)
        aggregated = _aggregate_router_logits(router_iterable, batch_size, seq_len, attention_mask)
        if aggregated is not None and aggregated.size(0) == len(slice_ids):
            aux_loss = aux_loss + 0.01 * balance_loss(aggregated)
            aux_loss = aux_loss + 0.01 * contrastive_loss(aggregated, slice_ids)

    total_loss = main_loss + aux_loss
    return (total_loss, outputs) if return_outputs else total_loss
