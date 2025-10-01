# Modified from https://github.com/huggingface/transformers/blob/bdf5fb70aa11782cce22027d76879f71f4e41c1e/src/transformers/models/qwen3_moe/modular_qwen3_moe.py

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
)
from transformers.utils.generic import OutputRecorder

from .functional import moe_fused_linear
from .kernels.indexing import get_expert_counts_and_idx


def moe_fused_kaiming_uniform_(weight: torch.Tensor) -> None:
    # Kaiming uniform on in_features
    # Although Qwen's default activation is silu, we set the gain `a = sqrt(5)` following the original Linear
    in_features = weight.shape[-1]
    bound = math.sqrt(3 * 5 / in_features)
    nn.init.uniform_(weight, -bound, bound)


class MoeFusedLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_experts"]
    in_features: int
    out_features: int
    num_experts: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty((num_experts, out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        moe_fused_kaiming_uniform_(self.weight)

    def forward(self, input: torch.Tensor, m_sizes: torch.Tensor) -> torch.Tensor:
        return moe_fused_linear(input, self.weight, m_sizes)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, num_experts={self.num_experts}"


# This class follows the implementation in HF Transformers
# patch_Qwen3MoeFusedSparseMoeBlock_forward can make it faster
class Qwen3MoeFusedSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_selected = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate_proj = MoeFusedLinear(self.hidden_size, self.moe_intermediate_size, config.num_experts)
        self.up_proj = MoeFusedLinear(self.hidden_size, self.moe_intermediate_size, config.num_experts)
        self.down_proj = MoeFusedLinear(self.moe_intermediate_size, self.hidden_size, config.num_experts)
        assert config.hidden_act == "silu"

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        M = batch_size * sequence_length

        hidden_states = hidden_states.view(M, hidden_dim)
        # router_logits: (M, num_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        # routing_weights, selected_experts: (M, num_selected)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_selected, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        hidden_states = hidden_states.unsqueeze(1).expand(M, self.num_selected, hidden_dim)
        # hidden_states must be contiguous
        hidden_states = hidden_states.reshape(M * self.num_selected, hidden_dim)
        selected_experts = selected_experts.view(M * self.num_selected)

        # Sort selected_experts and hidden_states for better memory coalescence of weight
        # It's possible to fuse a sort and a MoeFusedLinear layer, but for now we separate them for clarity
        m_sizes, sort_idx, inv_sort_idx = get_expert_counts_and_idx(selected_experts, self.num_experts)
        hidden_states = hidden_states[sort_idx]

        # It's possible to fuse gate_h and up_h, but this affects the shape of LoRA
        gate_h = self.gate_proj(hidden_states, m_sizes)
        up_h = self.up_proj(hidden_states, m_sizes)
        hidden_states = F.silu(gate_h) * up_h
        del gate_h, up_h
        hidden_states = self.down_proj(hidden_states, m_sizes)

        hidden_states = hidden_states[inv_sort_idx]

        hidden_states = hidden_states.view(M, self.num_selected, hidden_dim)
        hidden_states = torch.einsum("beo,be->bo", hidden_states, routing_weights)

        hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits


class Qwen3MoeFusedDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeFusedSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_outputs = self.mlp(hidden_states)
        router_logits = None
        if isinstance(mlp_outputs, tuple):
            hidden_states, router_logits = mlp_outputs
        else:
            hidden_states = mlp_outputs
        hidden_states = residual + hidden_states

        if output_router_logits:
            return hidden_states, router_logits
        return hidden_states


class Qwen3MoeFusedModel(Qwen3MoeModel):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3MoeFusedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output_router_logits = kwargs.pop("output_router_logits", self.config.output_router_logits)
        collected_router_logits: list[torch.Tensor] = []

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_router_logits:
                hidden_states, layer_router_logits = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    output_router_logits=True,
                    **kwargs,
                )
                if layer_router_logits is not None:
                    collected_router_logits.append(layer_router_logits)
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)

        router_logits_tuple = tuple(collected_router_logits) if collected_router_logits else None

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=router_logits_tuple,
        )


class Qwen3MoeFusedForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config)
        self.model = Qwen3MoeFusedModel(config)
        self._can_record_outputs["router_logits"] = OutputRecorder(Qwen3MoeFusedSparseMoeBlock, index=1)
        self._can_record_outputs["hidden_states"] = Qwen3MoeFusedDecoderLayer
