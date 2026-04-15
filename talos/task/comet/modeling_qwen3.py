#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/11/12 16:07:35
# Author: Shilei Liu
import json
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, can_return_tuple, logging

from talos.ops.linear_cross_entropy import chunked_linear_cross_entropy
from talos.task.comet.plugins import MemLayerPlugin, MemModelPlugin


__all__ = [
    "MemQwen3Config",
    "MemQwen3Layer",
    "MemQwen3PreTrainedModel",
    "MemQwen3Model",
    "MemQwen3ForCausalLM",
]


logger = logging.get_logger(__name__)

TensorOrTensors = Union[torch.Tensor, List[torch.Tensor]]


class MemQwen3Config(Qwen3Config):
    def __init__(
        self,
        *args,
        chunk_size: int = 512,
        placeholder_id: int = 128000,
        temp_beacon_id: int = 128001,
        temp_beacon_stride: int = 8,
        temp_mem_budget: int = 128,
        global_beacon_ids: List[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.placeholder_id = placeholder_id
        self.temp_beacon_id = temp_beacon_id
        self.temp_beacon_stride = temp_beacon_stride
        self.temp_mem_budget = temp_mem_budget
        self.global_beacon_ids = global_beacon_ids

    def __repr__(self):
        config_dict = self.to_diff_dict()
        for key in ["global_beacon_ids"]:
            val = config_dict.get(key)
            if val is not None:
                val = [str(t) for t in val]
                if len(val) > 16:
                    val = val[:6] + ["..."] + val[-6:]
                config_dict[key] = val
        json_str = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
        return f"{self.__class__.__name__} {json_str}"


@dataclass
class MemoryModelOutputWithPast(BaseModelOutputWithPast):
    global_memories: Optional[tuple[torch.FloatTensor]] = None
    global_states: Optional[tuple[torch.FloatTensor]] = None
    temp_memories: Optional[tuple[torch.FloatTensor]] = None


class Bottleneck(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x):
        return x + self.up_proj(self.down_proj(x))


class MemoryRecurrentNet(GradientCheckpointingLayer):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.g_proj = nn.Linear(2 * hidden_size, 1, bias=False)
        self.o_proj = Bottleneck(hidden_size, 8)

    def forward(self, current: torch.Tensor, state: Optional[torch.Tensor]):
        current = self.norm(current)
        if state is None:
            state = current
        else:
            g = self.g_proj(torch.cat([current, state], dim=-1)).sigmoid()
            state = g * state + (1 - g) * current
        o = self.o_proj(state)
        return o, state


class MemoryProjectionNet(GradientCheckpointingLayer, nn.Sequential):
    def __init__(self, hidden_size: int):
        super().__init__(nn.RMSNorm(hidden_size, eps=1e-12), Bottleneck(hidden_size, 8))


class MemQwen3Layer(nn.Module):
    def __init__(self, config: MemQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.inner_layer = Qwen3DecoderLayer(config, layer_idx)
        if len(config.global_beacon_ids) > 0:
            self.global_memory_net = MemoryRecurrentNet(config.hidden_size)
        else:
            self.global_memory_net = None
        if config.temp_beacon_stride != -1:
            self.temp_memory_net = MemoryProjectionNet(config.hidden_size)
        else:
            self.temp_memory_net = None
        self.temp_mem_budget = config.temp_mem_budget
        self.plugin = MemLayerPlugin(self.hidden_size, self.temp_mem_budget)
        self.attention_type = config.layer_types[layer_idx]
        assert self.attention_type == "full_attention"

    def set_layer_plugin(self, plugin):
        self.plugin = plugin

    def forward(
        self,
        hidden_states: TensorOrTensors,
        global_mem_inds: Optional[List[torch.Tensor]] = None,
        global_beacon_inds: Optional[List[torch.Tensor]] = None,
        temp_mem_inds: Optional[List[torch.Tensor]] = None,
        temp_beacon_inds: Optional[List[torch.Tensor]] = None,
        temp_mem_select_inds: Optional[List[torch.Tensor]] = None,
        global_memory: Optional[torch.Tensor] = None,
        global_state: Optional[torch.Tensor] = None,
        temp_memory: Optional[torch.Tensor] = None,
        attention_mask: Optional[TensorOrTensors] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[TensorOrTensors] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if isinstance(hidden_states, list):
            outputs = self.plugin(
                self.inner_layer,
                self.global_memory_net,
                self.temp_memory_net,
                hidden_states,
                global_mem_inds=global_mem_inds,
                global_beacon_inds=global_beacon_inds,
                temp_mem_inds=temp_mem_inds,
                temp_beacon_inds=temp_beacon_inds,
                temp_mem_select_inds=temp_mem_select_inds,
                global_memory=global_memory,
                global_state=global_state,
                temp_memory=temp_memory,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        else:
            outputs = self.inner_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return outputs


class MemQwen3PreTrainedModel(Qwen3PreTrainedModel):
    config_class = MemQwen3Config
    _no_split_modules = ["MemQwen3Layer"]
    _default_key_mapping = {
        r"(layers\.\d+)\.((self_attn|mlp|input_layernorm|post_attention_layernorm)\.)": r"\1.inner_layer.\2"
    }

    def set_plugin(self, plugin):
        for m in self.modules():
            if isinstance(m, MemQwen3Layer):
                m.set_layer_plugin(plugin)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if "key_mapping" not in kwargs:
            kwargs["key_mapping"] = cls._default_key_mapping
        return super().from_pretrained(*args, **kwargs)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (Qwen3RMSNorm, nn.RMSNorm)):
            module.weight.data.fill_(1.0)


class MemQwen3Model(MemQwen3PreTrainedModel):
    def __init__(self, config: MemQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                MemQwen3Layer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        # Does not support swa.
        assert not self.has_sliding_layers

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def build_memory_position_ids(self, position_ids: torch.Tensor, memory_size: int):
        device = position_ids.device
        start_positions = position_ids[:, -1].unsqueeze(1) + 1
        incremental_part = torch.arange(memory_size, device=device)
        memory_position_ids = start_positions + incremental_part
        return memory_position_ids

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        chunk_sizes: Optional[List[int]] = None,
        global_mem_inds: Optional[List[torch.LongTensor]] = None,
        global_beacon_inds: Optional[List[torch.LongTensor]] = None,
        temp_mem_inds: Optional[List[torch.Tensor]] = None,
        temp_beacon_inds: Optional[List[torch.Tensor]] = None,
        temp_mem_select_inds: Optional[List[torch.Tensor]] = None,
        global_memories: Optional[List[torch.Tensor]] = None,
        global_states: Optional[List[torch.Tensor]] = None,
        temp_memories: Optional[List[torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_memories: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if chunk_sizes is None:
            return Qwen3Model.forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        assert not output_attentions
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with training mode. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        is_prefill = True
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            assert isinstance(past_key_values, DynamicCache)
            if past_key_values.get_seq_length() > 0:
                is_prefill = False

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not is_prefill:
            attention_mask = attention_mask[:, sum(chunk_sizes[:-1]) :]

        if self.training and attention_mask is None:
            causal_mask = None
        else:
            causal_mask = create_causal_mask(
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

        if is_prefill:
            plugin = MemModelPlugin()
            chunked = plugin.chunk(
                hidden_states, position_embeddings, chunk_sizes, causal_mask
            )
            hidden_states_list_or_tensor = chunked.hidden_states
            attention_mask = chunked.attention_mask
            position_embeddings = chunked.position_embeddings
        else:
            hidden_states_list_or_tensor = hidden_states
            attention_mask = causal_mask

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_global_memories = () if output_memories else None
        all_global_states = () if output_memories else None
        all_temp_memories = () if output_memories else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if global_memories is not None and output_memories:
                global_memory = global_memories[i]
                global_state = global_states[i]
                temp_memory = temp_memories[i]
            else:
                global_memory, global_state, temp_memory = None, None, None

            layer_outputs = decoder_layer(
                hidden_states_list_or_tensor,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                global_mem_inds=global_mem_inds,
                global_beacon_inds=global_beacon_inds,
                temp_mem_inds=temp_mem_inds,
                temp_beacon_inds=temp_beacon_inds,
                temp_mem_select_inds=temp_mem_select_inds,
                global_memory=global_memory,
                global_state=global_state,
                temp_memory=temp_memory,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )

            hidden_states_list_or_tensor = layer_outputs[0]

            if output_attentions and not is_prefill:
                all_self_attns += (layer_outputs[1],)

            if is_prefill and output_memories:
                all_global_memories += (layer_outputs.global_memory,)
                all_global_states += (layer_outputs.global_state,)
                all_temp_memories += (layer_outputs.temp_memory,)

        if is_prefill:
            hidden_states = torch.cat(hidden_states_list_or_tensor, dim=1).contiguous()
        else:
            hidden_states = hidden_states_list_or_tensor

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MemoryModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            global_memories=all_global_memories,
            global_states=all_global_states,
            temp_memories=all_temp_memories,
        )

    def mem_effi_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        chunk_sizes: Optional[List[int]] = None,
        global_mem_inds: Optional[List[torch.LongTensor]] = None,
        global_beacon_inds: Optional[List[torch.LongTensor]] = None,
        temp_mem_inds: Optional[List[torch.Tensor]] = None,
        temp_beacon_inds: Optional[List[torch.Tensor]] = None,
        temp_mem_select_inds: Optional[List[torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        assert inputs_embeds is None
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        assert use_cache is True
        is_prefill = True
        if use_cache:
            if past_key_values is not None and past_key_values.get_seq_length() > 0:
                is_prefill = False
        if not is_prefill:
            return self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                chunk_sizes=chunk_sizes,
                global_mem_inds=global_mem_inds,
                global_beacon_inds=global_beacon_inds,
                temp_mem_inds=temp_mem_inds,
                temp_beacon_inds=temp_beacon_inds,
                temp_mem_select_inds=temp_mem_select_inds,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )
        num_chunks = len(chunk_sizes)
        input_ids_chunks = list(torch.split(input_ids, chunk_sizes, dim=1))
        position_ids_chunks = list(torch.split(position_ids, chunk_sizes, dim=1))
        if attention_mask is not None:
            attention_mask_chunks = list(
                torch.split(attention_mask, chunk_sizes, dim=1)
            )
        else:
            attention_mask_chunks = [None for _ in range(num_chunks)]
        if cache_position is not None:
            cache_position_chunks = [cache_position[:c] for c in chunk_sizes]
            cache_position_chunks = list(
                torch.split(cache_position, chunk_sizes, dim=0)
            )
        else:
            cache_position_chunks = [None for _ in range(num_chunks)]
        global_memories = [None for _ in range(len(self.layers))]
        global_states = [None for _ in range(len(self.layers))]
        temp_memories = [None for _ in range(len(self.layers))]
        for i in range(num_chunks):
            current_use_cache = False if i < num_chunks - 1 else use_cache
            output_memories = True if i < num_chunks - 1 else use_cache
            outputs: MemoryModelOutputWithPast = self.forward(
                input_ids=input_ids_chunks[i],
                attention_mask=attention_mask_chunks[i],
                position_ids=position_ids_chunks[i],
                chunk_sizes=[chunk_sizes[i]],
                global_mem_inds=[global_mem_inds[i]],
                global_beacon_inds=[global_beacon_inds[i]],
                temp_mem_inds=[temp_mem_inds[i]],
                temp_beacon_inds=[temp_beacon_inds[i]],
                temp_mem_select_inds=[temp_mem_select_inds[i]],
                global_memories=global_memories,
                global_states=global_states,
                temp_memories=temp_memories,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=current_use_cache,
                output_attentions=False,
                output_hidden_states=False,
                output_memories=output_memories,
                cache_position=cache_position_chunks[i],
                **flash_attn_kwargs,
            )
            global_memories = outputs.global_memories
            global_states = outputs.global_states
            temp_memories = outputs.temp_memories
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )


class MemQwen3ForCausalLM(MemQwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: MemQwen3Config):
        super().__init__(config)
        self.model = MemQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_type = "ForCausalLM"
        # Initialize weights and apply final processing
        self.post_init()
        self.register_buffer(
            "temp_beacon_id", torch.tensor([config.temp_beacon_id]), persistent=False
        )
        # For generation.
        self.cached_input_ids = None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        chunk_sizes: Optional[List[int]] = None,
        global_mem_inds: Optional[List[torch.LongTensor]] = None,
        global_beacon_inds: Optional[List[torch.LongTensor]] = None,
        temp_mem_inds: Optional[List[torch.Tensor]] = None,
        temp_beacon_inds: Optional[List[torch.Tensor]] = None,
        temp_mem_select_inds: Optional[List[torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mem_effi: Optional[bool] = False,
        last_chunk_text_length: Optional[int] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        if mem_effi:
            assert labels is None
            assert not self.training
            forward_method = self.model.mem_effi_forward
        else:
            forward_method = self.model

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = forward_method(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            chunk_sizes=chunk_sizes,
            global_mem_inds=global_mem_inds,
            global_beacon_inds=global_beacon_inds,
            temp_mem_inds=temp_mem_inds,
            temp_beacon_inds=temp_beacon_inds,
            temp_mem_select_inds=temp_mem_select_inds,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        loss, logits = None, None

        if labels is not None:
            labels = labels.view(-1)

            num_items_in_batch = (labels != -100).sum()
            dist.all_reduce(num_items_in_batch, op=dist.ReduceOp.AVG)
            num_items_in_batch = num_items_in_batch.clamp(min=1)

            if isinstance(self.lm_head, FSDPModule):
                self.lm_head.unshard()
            weight = self.lm_head.weight
            d_model = hidden_states.shape[-1]
            hidden_states = hidden_states.view(-1, d_model)

            loss = chunked_linear_cross_entropy(
                hidden_states,
                weight,
                labels,
                n_loop_iters=8,
                ignore_index=-100,
                reduction="sum",
            )
            loss = loss / num_items_in_batch
        else:
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        if model_kwargs.get("chunk_sizes") is not None:
            position_ids = model_kwargs["position_ids"]
            new_position_ids = position_ids[:, -1:] + 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_ids], dim=-1
            )

            last_chunk_text_length = model_kwargs["last_chunk_text_length"]
            if model_kwargs.get("need_insert_beacon", False):
                outputs.logits[:, :, self.config.temp_beacon_id] = 10000
            else:
                model_kwargs["last_chunk_text_length"] += 1

            model_kwargs["need_insert_beacon"] = (
                self.config.temp_beacon_stride > 0
                and last_chunk_text_length % self.config.temp_beacon_stride == 0
            )
        return super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        if "position_ids" not in kwargs and kwargs.get("chunk_sizes") is not None:
            raise ValueError("`position_ids` is a required argument.")
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            **kwargs,
        )
        if kwargs.get("chunk_sizes") is not None:
            model_inputs["mem_effi"] = kwargs.get("mem_effi", False)
            model_inputs["chunk_sizes"] = kwargs["chunk_sizes"]
            model_inputs["global_mem_inds"] = kwargs["global_mem_inds"]
            model_inputs["global_beacon_inds"] = kwargs["global_beacon_inds"]
            model_inputs["temp_mem_inds"] = kwargs["temp_mem_inds"]
            model_inputs["temp_beacon_inds"] = kwargs["temp_beacon_inds"]
            if "need_insert_beacon" not in kwargs:
                self.cached_input_ids = None
            elif kwargs["need_insert_beacon"]:
                self.cached_input_ids = model_inputs["input_ids"]
                batch_size = model_inputs["input_ids"].shape[0]
                beacons = self.temp_beacon_id.unsqueeze(0).repeat(batch_size, 1)
                model_inputs["input_ids"] = beacons
            elif self.cached_input_ids is not None:
                model_inputs["input_ids"] = self.cached_input_ids
                self.cached_input_ids = None
        return model_inputs
