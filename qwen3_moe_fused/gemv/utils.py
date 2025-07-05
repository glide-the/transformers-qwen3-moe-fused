from functools import partial

import torch


@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad
def get_batch_begins_ends(s: torch.Tensor, E: int) -> torch.Tensor:
    arange = torch.arange(E, device=s.device, dtype=s.dtype)
    s_begins = (arange[:, None] > s[None, :]).to(torch.int32).sum(dim=1)
    s_ends = (arange[:, None] >= s[None, :]).to(torch.int32).sum(dim=1)
    s_begins_ends = torch.stack([s_begins, s_ends], dim=1)
    return s_begins_ends


@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad()
def get_routing_indices(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
    token_counts_by_expert = torch.histc(
        selected_experts.view(-1),
        bins=num_experts,
        min=0,
        max=num_experts,
    )
    return token_counts_by_expert
