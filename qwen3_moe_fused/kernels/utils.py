from functools import partial

import torch


@partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
@torch.no_grad
def compute_batch_begins_ends(s: torch.Tensor, E: int) -> torch.Tensor:
    arange = torch.arange(E, device=s.device, dtype=s.dtype)
    s_begins = (arange[:, None] > s[None, :]).to(torch.int32).sum(dim=1)
    s_ends = (arange[:, None] >= s[None, :]).to(torch.int32).sum(dim=1)
    s_begins_ends = torch.stack([s_begins, s_ends], dim=1)
    return s_begins_ends
