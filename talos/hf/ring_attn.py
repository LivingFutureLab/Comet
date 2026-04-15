#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/11/23 22:16:42
# Author: Shilei Liu
from typing import Optional

import torch
import torch.distributed as dist


__all__ = ["te_attention_forward", "te_attention_patch"]


def te_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    def _transpose(t):
        return t.transpose(1, 2).contiguous()

    query = _transpose(query)
    key = _transpose(key)
    value = _transpose(value)
    attn_output = module.attention(query, key, value)
    return attn_output, None


def te_attention_patch(self, config, cp_group, cp_global_ranks, cp_stream):
    import transformer_engine.pytorch as te

    self.cp_group = cp_group
    self.cp_global_ranks = cp_global_ranks
    self.cp_stream = cp_stream
    self.cp_size = dist.get_world_size(cp_group)
    self.attention = te.DotProductAttention(
        num_attention_heads=config.num_attention_heads,
        kv_channels=self.head_dim,
        num_gqa_groups=config.num_key_value_heads,
        attention_dropout=self.attention_dropout,
        attn_mask_type="causal",
        qkv_format="bshd",
        layer_number=self.layer_idx,
        cp_group=cp_group,
        cp_global_ranks=cp_global_ranks,
        cp_stream=cp_stream,
        cp_comm_type="p2p",
    )
