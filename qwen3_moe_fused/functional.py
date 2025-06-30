import torch


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


# TODO: Write a Triton kernel to avoid allocating an array of shape (b, o, i)
# TODO: Test if rearranging the layouts can improve cache hit
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


moe_fused_linear = moe_fused_linear_torch


def _test():
    from math import sqrt

    batch_size = 2
    in_features = 3
    out_features = 5
    num_experts = 7
    device = "cuda"
    dtype = torch.float32

    input = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    weight = 1 / sqrt(in_features) * torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (batch_size,), device=device, dtype=torch.int32)

    output_naive = moe_fused_linear_naive(input, weight, selected_experts)

    output_torch = moe_fused_linear_torch(input, weight, selected_experts)
    print(torch.allclose(output_torch, output_naive, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    _test()
