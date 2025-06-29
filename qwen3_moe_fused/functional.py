import torch


def moe_fused_linear_naive(
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_selected, in_features = input.shape
    num_experts, out_features, _ = weight.shape

    output = torch.empty(batch_size, num_selected, out_features, device=input.device, dtype=input.dtype)
    for b in range(batch_size):
        for e in range(num_selected):
            expert_idx = selected_experts[b, e].item()
            _weight = weight[expert_idx, :, :]
            _input = input[b, e, :]
            output[b, e, :] = _weight @ _input
    return output


# TODO: Write a Triton kernel to avoid allocating an array of shape (b * e, o, i)
# TODO: Test if rearranging the layouts can improve cache hit
def moe_fused_linear_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a MoE linear operation using vectorized operations.

    The operation is defined as:
    `output[b, e, o] = sum_i weight[selected_experts[b, e], o, i] * input[b, e, i]`

    Args:
        input (`torch.FloatTensor`): input tensor of shape `(batch_size, num_selected, in_features)`.
        weight (`torch.FloatTensor`): weight tensor of shape `(num_experts, out_features, in_features)`.
        selected_experts (`torch.LongTensor`): tensor of selected expert indices in shape `(batch_size, num_selected)`.
            Each element is in the range `[0, num_experts)`.

    Returns:
        output (`torch.FloatTensor`): output tensor of shape `(batch_size, num_selected, out_features)`.
    """
    batch_size, num_selected, in_features = input.shape
    num_experts, out_features, _ = weight.shape
    M = batch_size * num_selected

    input = input.view(M, in_features)
    selected_experts = selected_experts.view(M)
    weight = weight[selected_experts]
    output = torch.einsum("moi,mi->mo", weight, input)
    output = output.view(batch_size, num_selected, out_features)
    return output


moe_fused_linear = moe_fused_linear_torch


def _test():
    from math import sqrt

    batch_size = 2
    in_features = 3
    out_features = 5
    num_selected = 7
    num_experts = 11
    device = "cuda"
    dtype = torch.float32

    input = torch.randn(batch_size, num_selected, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (batch_size, num_selected), device=device, dtype=torch.int32)

    output_naive = moe_fused_linear_naive(input, weight, selected_experts)

    output_torch = moe_fused_linear_torch(input, weight, selected_experts)
    print(torch.allclose(output_torch, output_naive, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    _test()
