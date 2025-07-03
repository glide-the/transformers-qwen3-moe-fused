Work in progress. Still needs a lot of optimization.

The purpose is to fine-tune Qwen3-30B-A3B on a single GPU with 24GB VRAM and achieve high throughput. The implementation is compatible with the HF Transformers ecosystem, such as LoRA, bitsandbytes 4-bit quantization, and Unsloth.

The critical part is to implement the [`moe_fused_linear`](https://github.com/woct0rdho/transformers-qwen3-moe-fused/blob/master/qwen3_moe_fused/functional.py) function:
```
output[b, o] = sum_i weight[selected_experts[b], o, i] * input[b, i]
```
This is the same as `MUL_MAT_ID` in llama.cpp .
