#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/12/20 23:32:19
# Author: Shilei Liu
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.autograd.profiler import record_function
from transformers.cache_utils import Cache

from talos.task.comet.distributed.p2p import recv_with_grad, send_with_grad
from talos.task.comet.distributed.parallel_state import ParallelState
from talos.task.comet.plugins import MemLayerOutputs, MemLayerPlugin


__all__ = ["MemLayerContextParallelismPlugin"]


class MemLayerContextParallelismPlugin(MemLayerPlugin):
    def __init__(
        self,
        hidden_size: int,
        temp_mem_budget: int,
        parallel_state: ParallelState,
    ):
        super().__init__(hidden_size, temp_mem_budget)
        self.parallel_state = parallel_state

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
        is_training: bool = True,
    ):
        group = self.parallel_state.group
        rank = self.parallel_state.rank
        is_first_rank = self.parallel_state.is_first_rank
        is_last_rank = self.parallel_state.is_last_rank

        if not is_first_rank:
            dtype = hidden_states[0].dtype
            device = hidden_states[0].device
            hidden_size = hidden_states[0].shape[-1]

            gm_shape = (*global_mem_inds[0].shape, hidden_size)
            gs_shape = gm_shape
            tm_shape = (*temp_mem_inds[0].shape, hidden_size)

            with record_function("## Recv memory ##"):
                if global_mem_net is not None:
                    global_memory = recv_with_grad(
                        gm_shape, dtype, device, src=rank - 1, group=group
                    )
                    global_state = recv_with_grad(
                        gs_shape, dtype, device, src=rank - 1, group=group
                    )
                if tm_shape[1] != 0:
                    temp_memory = recv_with_grad(
                        tm_shape, dtype, device, src=rank - 1, group=group
                    )

        outputs: MemLayerOutputs = super().__call__(
            forward_func,
            global_mem_net,
            temp_mem_net,
            hidden_states,
            global_mem_inds,
            global_beacon_inds,
            temp_mem_inds,
            temp_beacon_inds,
            temp_mem_select_inds,
            global_memory,
            global_state,
            temp_memory,
            attention_mask,
            position_embeddings,
            past_key_value,
            use_cache,
            cache_position,
        )

        gm = outputs.global_memory
        gs = outputs.global_state
        tm = outputs.temp_memory

        if not is_last_rank:
            with record_function("## Send memory ##"):
                if global_mem_net is not None:
                    gm = send_with_grad(gm, rank + 1, group)
                    gs = send_with_grad(gs, rank + 1, group)
                if tm is not None:
                    tm = send_with_grad(tm, rank + 1, group)

        if is_training:
            # HACK: Graft the gradient path from memories to the main computation graph.
            # In this rank, the generated `short_term_memory` and `long_term_memory`
            # are only sent to the next rank and are not used in any local computation
            # that contributes to the final loss. As a result, PyTorch's autograd
            # would not normally propagate gradients back to them, breaking the
            # backward pass of `send_with_grad` and causing a deadlock.
            #
            # To solve this, we create a "dummy" dependency. We add a zero-valued
            # tensor derived from `stm` and `ltm` to the last chunk of hidden_states.
            # This operation does not change the forward pass values but ensures that
            # a computational path exists, allowing gradients to flow back to `stm`
            # and `ltm`, thus triggering the backward hooks of `send_with_grad`.
            grad_connector_g = gm.sum() + gs.sum() if global_mem_net is not None else 0
            grad_connector_t = tm.sum() if tm is not None else 0
            grad_connector = 0.0 * (grad_connector_g + grad_connector_t).view(1, 1, 1)
            hidden_states = outputs.hidden_states[-1]
            outputs.hidden_states[-1] = hidden_states + grad_connector

        return outputs
