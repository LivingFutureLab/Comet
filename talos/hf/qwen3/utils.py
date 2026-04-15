#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/11/30 14:26:59
# Author: Shilei Liu
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


__all__ = ["qwen3_train_tflops"]


def qwen3_train_tflops(config: "Qwen3Config", seq_len: int, qps: float) -> float:
    """Calculate train tflops.
    Ignore communication, embedding lookup, rms norm, softmax and grad recompute.

    Args:
        seq_len (int): Sequence length.
        qps (float): Consumed train samples per second.

    Returns:
        int: Approximate train tflops.
    """
    d = config.hidden_size
    im = config.intermediate_size
    l = config.num_hidden_layers  # noqa: E741
    v = config.vocab_size
    nh = config.num_attention_heads
    nkvh = config.num_key_value_heads
    dh = config.head_dim
    da = nh * dh

    mlp_flops = 6 * seq_len * d * im
    qo_flops = 4 * seq_len * d * da
    kv_flops = 4 * seq_len * d * nkvh * dh
    qkvo_flops = qo_flops + kv_flops
    attn_flops = 4 * seq_len**2 * da
    logits_flops = 2 * seq_len * d * v
    forward_flops = qps * (l * (mlp_flops + qkvo_flops + attn_flops) + logits_flops)
    backward_flops = forward_flops * 2
    approx_train_flops = forward_flops + backward_flops
    return approx_train_flops / 1e12
