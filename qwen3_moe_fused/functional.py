import torch

from .kernels.index_matmul import index_matmul
from .kernels.index_matmul_sorted import index_matmul_sorted


# Reference implementation of index_matmul
def _moe_fused_linear_naive_fwd(
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

    output = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    for b in range(batch_size):
        _weight = weight[selected_experts[b], :, :]
        _input = input[b, :]
        output[b, :] = _weight @ _input
    return output


# Reference implementation of index_matmul_sorted
# Sort selected_experts for better memory coalescence of weight
def _moe_fused_linear_naive_sorted_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    selected_experts, sort_idx = torch.sort(selected_experts)
    output = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    for b in range(batch_size):
        _weight = weight[selected_experts[b], :, :]
        _input = input[sort_idx[b], :]
        output[sort_idx[b], :] = _weight @ _input
    return output


def _moe_fused_linear_naive_bwd(
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


# Vectorized version of _moe_fused_linear_naive_fwd,
# but it allocates weight_selected (batch_size, out_features, in_features) and takes too much memory
def _moe_fused_linear_torch_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    weight_selected = weight[selected_experts]
    output = torch.einsum("boi,bi->bo", weight_selected, input).to(input.dtype)
    return output


def _moe_fused_linear_torch_bwd(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    batch_size, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    # grad_input[b, i] = sum_o weight[selected_experts[b], o, i] * grad_output[b, o]
    weight_selected = weight[selected_experts]
    grad_input = torch.einsum("bo,boi->bi", grad_output, weight_selected).to(input.dtype)

    # for b in range(batch_size):
    #     grad_weight[selected_experts[b], o, i] += grad_output[b, o] * input[b, i]
    grad_weight_selected = torch.einsum("bo,bi->boi", grad_output, input).to(weight.dtype)
    idx = selected_experts.to(torch.int64).view(batch_size, 1, 1).expand(-1, out_features, in_features)
    grad_weight = torch.zeros_like(weight)
    grad_weight.scatter_add_(0, idx, grad_weight_selected)

    return grad_input, grad_weight, None


# After compiling, they do not take too much memory
# no-cudagraphs is needed for autograd
_moe_fused_linear_torch_fwd_compiled = torch.compile(
    _moe_fused_linear_torch_fwd, fullgraph=True, mode="max-autotune-no-cudagraphs"
)
_moe_fused_linear_torch_bwd_compiled = torch.compile(
    _moe_fused_linear_torch_bwd, fullgraph=True, mode="max-autotune-no-cudagraphs"
)

_moe_fused_linear_triton_fwd = index_matmul
_moe_fused_linear_triton_sorted_fwd = index_matmul_sorted


# If we do autograd on the compiled forward function, then the backward function will not be compiled and will still
# take too much memory. We can compile the backward function again, but for convenience we define a custom autograd
# function to ensure that the backward function is compiled
class MoeFusedLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, selected_experts):
        ctx.save_for_backward(input, weight, selected_experts)
        if input.is_cuda:
            # In Qwen3MoeFusedSparseMoeBlock, we do the sort outside the 3 MoeFusedLinear modules,
            # so we don't do the sort here
            return _moe_fused_linear_triton_fwd(input, weight, selected_experts)
        else:
            return _moe_fused_linear_torch_fwd_compiled(input, weight, selected_experts)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, selected_experts = ctx.saved_tensors
        return _moe_fused_linear_torch_bwd_compiled(grad_output, input, weight, selected_experts)


moe_fused_linear = MoeFusedLinearFunc.apply
