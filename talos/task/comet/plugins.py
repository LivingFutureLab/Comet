#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/04 19:36:20
# Author: Shilei Liu
# Description: Common tools for unified memory transformer.
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.utils.generic import ModelOutput


__all__ = ["MemLayerPlugin", "MemModelPlugin"]


@dataclass
class ChunkedInputs(ModelOutput):
    """
    A data structure to hold the input tensors after they have been split into chunks.
    Each attribute is a list of tensors, where each tensor corresponds to a single chunk.
    """

    hidden_states: Optional[List[torch.Tensor]] = None
    position_embeddings: Optional[List[torch.Tensor]] = None
    attention_mask: Optional[List[torch.Tensor]] = None


@dataclass
class MemLayerOutputs(ModelOutput):
    hidden_states: Optional[List[torch.Tensor]] = None
    global_memory: Optional[torch.Tensor] = None
    global_state: Optional[torch.Tensor] = None
    temp_memory: Optional[torch.Tensor] = None
    others: Optional[Tuple[List[Any]]] = None


class MemLayerPlugin:
    def __init__(self, hidden_size: int, temp_mem_budget: int):
        self.hidden_size = hidden_size
        self.temp_mem_budget = temp_mem_budget

    def inject_memory(
        self, hidden_states: torch.Tensor, indices: torch.Tensor, memory: torch.Tensor
    ):
        hidden_size = hidden_states.shape[2]
        indices = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        outputs = hidden_states.scatter(1, indices, memory)
        return outputs

    def extract_memory(self, hidden_states: torch.Tensor, indices: torch.Tensor):
        hidden_size = hidden_states.shape[2]
        indices = indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        outputs = hidden_states.gather(1, indices)
        return outputs

    def memory_enqueue(
        self,
        slots: torch.Tensor,
        queue: Optional[torch.Tensor],
        select_inds: torch.Tensor,
    ):
        queue = torch.cat([queue, slots], dim=1) if queue is not None else slots
        batch_size = queue.shape[0]
        batch_inds = torch.arange(batch_size, device=queue.device).unsqueeze(1)
        queue = queue[batch_inds, select_inds]
        return queue

    def __call__(
        self,
        forward_func: Callable[[Any], torch.Tensor],
        global_mem_net: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        temp_mem_net: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: List[torch.Tensor],
        global_mem_inds: List[torch.Tensor] = None,
        global_beacon_inds: List[torch.Tensor] = None,
        temp_mem_inds: List[torch.Tensor] = None,
        temp_beacon_inds: List[torch.Tensor] = None,
        temp_mem_select_inds: List[torch.Tensor] = None,
        global_memory: Optional[torch.Tensor] = None,
        global_state: Optional[torch.Tensor] = None,
        temp_memory: Optional[torch.Tensor] = None,
        attention_mask: Optional[List[torch.Tensor]] = None,
        position_embeddings: Optional[List[torch.Tensor]] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        num_chunks = len(hidden_states)

        outputs = []
        others = []

        for i in range(num_chunks):
            cur_hidden_states = hidden_states[i]

            cur_global_mem_inds = global_mem_inds[i]
            cur_global_beacon_inds = global_beacon_inds[i]

            cur_temp_mem_inds = temp_mem_inds[i]
            cur_temp_beacon_inds = temp_beacon_inds[i]

            cur_temp_mem_select_inds = None
            if temp_mem_select_inds is not None:
                cur_temp_mem_select_inds = temp_mem_select_inds[i]

            # Inject global memory from the previous chunk.
            if global_memory is not None:
                assert (
                    cur_global_mem_inds.numel()
                    == global_memory.numel() // self.hidden_size
                )
                cur_hidden_states = self.inject_memory(
                    cur_hidden_states, cur_global_mem_inds, global_memory
                )

            # Inject temp memory accumulated from all previous chunks.
            if temp_memory is not None:
                cur_hidden_states = self.inject_memory(
                    cur_hidden_states, cur_temp_mem_inds, temp_memory
                )

            if attention_mask is not None:
                cur_attention_mask = attention_mask[i]
            else:
                cur_attention_mask = None

            cur_position_embeddings = position_embeddings[i]

            # Pass KV cache arguments only to the last chunk for generation.
            kwargs = {}
            if use_cache and i == num_chunks - 1:
                kwargs["past_key_value"] = past_key_value
                kwargs["use_cache"] = use_cache
                kwargs["cache_position"] = cache_position

            # Call the original layer's forward pass.
            inner_outputs = forward_func(
                cur_hidden_states,
                attention_mask=cur_attention_mask,
                position_embeddings=cur_position_embeddings,
                **kwargs,
            )
            cur_output_hiddens: torch.Tensor = inner_outputs[0]
            others.append(inner_outputs[1:])
            outputs.append(cur_output_hiddens)

            # Extract and project the new short-term memory for the next chunk.
            if cur_global_beacon_inds.numel() > 0:
                cur_mem = self.extract_memory(
                    cur_output_hiddens, cur_global_beacon_inds
                )
                global_memory, global_state = global_mem_net(cur_mem, global_state)

            # Extract, project, and update the long-term memory.
            if cur_temp_beacon_inds.numel() > 0:
                cur_mem = self.extract_memory(cur_output_hiddens, cur_temp_beacon_inds)
                cur_mem = temp_mem_net(cur_mem)

                temp_memory = self.memory_enqueue(
                    cur_mem, temp_memory, cur_temp_mem_select_inds
                )

        others = tuple(map(list, zip(*others)))
        return MemLayerOutputs(
            hidden_states=outputs,
            global_memory=global_memory,
            global_state=global_state,
            temp_memory=temp_memory,
            others=others,
        )


class MemModelPlugin:
    def chunk(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        chunk_sizes: List[int],
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Splits flat, batch-level input tensors into a list of chunk-specific tensors.

        This method acts as a pre-processor for the model's forward pass. It takes the
        single large tensors that represent the concatenated chunks and slices them
        according to `chunk_sizes`. This prepares the inputs for the sequential,
        chunk-by-chunk processing performed by `MemLayerPlugin`.

        Args:
            hidden_states (torch.Tensor): The main input hidden states for the entire batch.
            position_embeddings (torch.Tensor): The corresponding position embeddings (e.g., RoPE).
            chunk_sizes (List[int]): A list of sizes for each chunk in the sequence.
            attention_mask (Optional[torch.Tensor]): The 4D attention mask for the entire batch.

        Returns:
            ChunkedInputs: An object containing lists of tensors, where each list element
            corresponds to a single chunk.
        """
        hidden_states_list = []
        position_embeddings_list = []
        if attention_mask is not None:
            attention_mask_list = []
        else:
            attention_mask_list = None

        beg = 0
        for chunk_size in chunk_sizes:
            end = beg + chunk_size
            hidden_states_list.append(hidden_states[:, beg:end, :].contiguous())
            # Handle RoPE position embeddings (cos, sin tuple)
            cos, sin = position_embeddings
            cos_chunk = cos[:, beg:end, :]
            sin_chunk = sin[:, beg:end, :]
            position_embeddings_list.append((cos_chunk, sin_chunk))

            if attention_mask_list is not None:
                # Slice the 4D attention mask for the current chunk
                cur_attention_mask = attention_mask[:, :, beg:end, beg:end]
                attention_mask_list.append(cur_attention_mask)
            beg = end

        return ChunkedInputs(
            hidden_states_list, position_embeddings_list, attention_mask_list
        )
