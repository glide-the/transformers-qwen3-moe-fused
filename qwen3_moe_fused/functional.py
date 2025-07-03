import torch

from .kernels.index_matmul import index_matmul


def moe_fused_linear_naive(
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    output = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    for b in range(batch_size):
        _weight = weight[selected_experts[b], :, :]
        _input = input[b, :]
        output[b, :] = _weight @ _input
    return output


def moe_fused_linear_naive_bwd(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    grad_input = torch.empty_like(input)
    for b in range(batch_size):
        _weight = weight[selected_experts[b], :, :]
        _grad_output = grad_output[b, :]
        grad_input[b, :] = _grad_output @ _weight

    grad_weight = torch.zeros_like(weight)
    for b in range(batch_size):
        grad_weight[selected_experts[b], :, :] += grad_output[b, :, None] * input[b, None, :]

    return grad_input, grad_weight, None


def moe_fused_linear_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a MoE linear operation using vectorized operations.

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

    weight_selected = weight[selected_experts]
    output = torch.einsum("boi,bi->bo", weight_selected, input)
    return output


def moe_fused_linear_torch_bwd(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    # grad_input[b, i] = sum_o weight[selected_experts[b], o, i] * grad_output[b, o]
    weight_selected = weight[selected_experts]
    grad_input = torch.einsum("bo,boi->bi", grad_output, weight_selected)

    # for b in range(batch_size):
    #     grad_weight[selected_experts[b], o, i] += grad_output[b, o] * input[b, i]
    grad_weight_selected = torch.einsum("bo,bi->boi", grad_output, input)
    idx = selected_experts.to(torch.int64).view(batch_size, 1, 1).expand(-1, out_features, in_features)
    grad_weight = torch.zeros_like(weight)
    grad_weight.scatter_add_(0, idx, grad_weight_selected)

    return grad_input, grad_weight, None


moe_fused_linear_triton = index_matmul
moe_fused_linear = moe_fused_linear_triton
