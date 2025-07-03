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

    weight = weight[selected_experts]
    output = torch.einsum("boi,bi->bo", weight, input)
    return output


moe_fused_linear_triton = index_matmul
moe_fused_linear = moe_fused_linear_triton
