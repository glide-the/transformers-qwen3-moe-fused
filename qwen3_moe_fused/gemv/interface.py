import torch

from .index_matmul import index_matmul
from .index_matmul_transposed import index_matmul_transposed
from .matmul_scatter_add import matmul_scatter_add


class MoeFusedLinearGemv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, selected_experts):
        ctx.save_for_backward(input, weight, selected_experts)
        return index_matmul(input, weight, selected_experts)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, selected_experts = ctx.saved_tensors
        grad_input = index_matmul_transposed(grad_output, weight, selected_experts, input.dtype)
        grad_weight = matmul_scatter_add(input, grad_output, selected_experts, weight.shape[0], weight.dtype)
        return grad_input, grad_weight, None


moe_fused_linear_gemv = MoeFusedLinearGemv.apply
