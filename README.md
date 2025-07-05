# Qwen3 MoE Fused

The purpose of this repo is to fine-tune Qwen3-30B-A3B on a single GPU with 24GB VRAM and achieve high throughput. The implementation is compatible with the HF Transformers ecosystem, such as LoRA, bitsandbytes 4-bit quantization, and Unsloth. See [`example_train_30b_a3b_unsloth.py`](https://github.com/woct0rdho/transformers-qwen3-moe-fused/blob/master/example_train_30b_a3b_unsloth.py) for the usage.

The critical part is to implement the [`moe_fused_linear`](https://github.com/woct0rdho/transformers-qwen3-moe-fused/blob/master/qwen3_moe_fused/functional.py) function:
```
output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]
```
There have been several high-performance implementation of this, such as [Unsloth](https://github.com/unslothai/unsloth/blob/2bfc39b6387577457834059c59f83fcdb954c9bd/unsloth/kernels/moe/grouped_gemm/kernels/forward.py), [triton-kernels](https://github.com/triton-lang/triton/blob/dd1c3d429d1c24904722ac699ea5750bc694c4d6/python/triton_kernels/triton_kernels/matmul_ogs.py), [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/a0535ffa0d35fccfec3e1a0a3bfc9dbb6054d7c0/ggml/src/ggml-cuda/ggml-cuda.cu#L2065), [vllm](https://github.com/vllm-project/vllm/blob/015fab8c2fa4db8776f7e91abd50371911673d88/vllm/model_executor/layers/fused_moe/fused_moe.py).
