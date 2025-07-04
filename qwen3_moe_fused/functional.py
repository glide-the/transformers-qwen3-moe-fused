from functools import partial

import torch

from .grouped_gemm.interface import grouped_gemm


def _moe_fused_linear_naive_fwd(
    input: torch.Tensor, weight: torch.Tensor, selected_experts: torch.Tensor
) -> torch.Tensor:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    output = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    for b in range(batch_size):
        _weight = weight[selected_experts[b], :, :]
        _input = input[b, :]
        output[b, :] = _weight @ _input
    return output


def _moe_fused_linear_naive_bwd_input(
    grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, selected_experts: torch.Tensor
) -> torch.Tensor:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    grad_input = torch.empty_like(input)
    for b in range(batch_size):
        _weight = weight[selected_experts[b], :, :]
        _grad_output = grad_output[b, :]
        grad_input[b, :] = _grad_output @ _weight
    return grad_input


def _moe_fused_linear_naive_bwd_weight(
    grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, selected_experts: torch.Tensor
) -> torch.Tensor:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    grad_weight = torch.zeros_like(weight)
    for b in range(batch_size):
        grad_weight[selected_experts[b], :, :] += grad_output[b, :, None] * input[b, None, :]
    return grad_weight


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


def _moe_fused_linear_grouped_gemm_fwd(
    input: torch.Tensor, weight: torch.Tensor, selected_experts: torch.Tensor
) -> torch.Tensor:
    """
    Computes a MoE linear operation using grouped GEMM in Triton.

    The operation is defined as:
    `output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]`

    Args:
        input (`torch.FloatTensor`): input tensor of shape `(batch_size, in_features)`.
        weight (`torch.FloatTensor`): weight tensor of shape `(num_experts, out_features, in_features)`.
        selected_experts (`torch.LongTensor`): tensor of selected expert indices in shape `(batch_size,)`.
            Each element is in the range `[0, num_experts)`.

    Returns:
        output (`torch.FloatTensor`): output tensor of shape `(batch_size, out_features)`.
    """
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    token_counts_by_expert = get_routing_indices(selected_experts, num_experts)
    return grouped_gemm(
        X=input,
        W=weight,
        m_sizes=token_counts_by_expert,
        topk=1,  # Not used
        autotune=True,
    )


class MoeFusedLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, selected_experts):
        ctx.save_for_backward(input, weight, selected_experts)
        return _moe_fused_linear_grouped_gemm_fwd(input, weight, selected_experts)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, selected_experts = ctx.saved_tensors
        # TODO
        # grad_input = _moe_fused_linear_triton_bwd_input(grad_output, input, weight, selected_experts)
        # grad_weight = _moe_fused_linear_triton_bwd_weight(grad_output, input, weight, selected_experts)
        # return grad_input, grad_weight, None


moe_fused_linear = MoeFusedLinearFunc.apply
