#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/15 17:01:18
# Author: Shilei Liu
from functools import partial
from typing import Any, List, Optional

import torch
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel


__all__ = ["get_num_parameters", "fsdp_wrap_hf_model", "GLUExpertWeightFuser"]


def fsdp_wrap_hf_model(
    model: PreTrainedModel,
    fsdp_kwargs: Optional[dict[str, Any]] = None,
    recompute: bool = False,
):
    """
    Wrap a Hugging Face `PreTrainedModel` with PyTorch FSDP (Fully Sharded Data Parallel v2).

    This function applies `torch.distributed.fsdp.fully_shard` to various parts of
    the Transformer model, including:
      - Non-split modules specified by `model._no_split_modules`.
      - Token embedding layers.
      - Lm_head layers.
      - The entire model itself.

    Optionally, it can also enable activation checkpointing for layers of type
    `GradientCheckpointingLayer` to trade compute for memory.

    Args:
        model (PreTrainedModel):
            The Hugging Face Transformer model to be wrapped with FSDP.
        fsdp_kwargs (Optional[dict[str, Any]]):
            Keyword arguments passed to `torch.distributed.fsdp.fully_shard`.
            Example keys: `{'reshard_after_forward': True, 'mp_policy': ...}`.
        recompute (bool):
            If True, apply activation checkpointing to eligible layers
            (layers that are instances of `GradientCheckpointingLayer`).

    Returns:
        None: This function wraps the given model in-place.
    """

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp import fully_shard

    fsdp_kwargs = {} if fsdp_kwargs is None else fsdp_kwargs

    # Wrap layers.
    for _, module in model.named_modules():
        if type(module).__name__.lstrip("FSDP") in model._no_split_modules:
            fully_shard(module, **fsdp_kwargs)

    # Wrap world embedding look up table.
    fully_shard(model.get_input_embeddings(), **fsdp_kwargs)

    # Wrap lm_head.
    if not model.config.tie_word_embeddings and hasattr(model, "get_output_embeddings"):
        fully_shard(model.get_output_embeddings(), **fsdp_kwargs)

    # Wrap whole model.
    fully_shard(model, **fsdp_kwargs)

    if recompute:

        def check_fn(m):
            return isinstance(m, GradientCheckpointingLayer)

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )


def get_num_parameters(model: PreTrainedModel) -> dict[str, str]:
    """
    Count the number of parameters in a given Hugging Face PreTrainedModel,
    and return a dictionary with comma-separated string formatting.

    This function calculates:
      1. Total parameter count (all layers).
      2. Backbone (non-embedding) parameter count,
         which excludes:
           - Input token embedding parameters.
           - Output embedding / lm_head parameters.

    Args:
        model (PreTrainedModel): A Hugging Face transformer model instance.

    Returns:
        dict[str, str]:
            {
                "total": "<formatted total count, e.g., '7,123,456'>",
                "backbone": "<formatted backbone count>"
            }
    """
    n_total = sum(p.numel() for p in model.parameters())
    n_embed = model.get_input_embeddings().weight.numel()
    if model.config.tie_word_embeddings:
        n_lm_head = 0
    else:
        n_lm_head = model.get_output_embeddings().weight.numel()
    n_backbone = n_total - n_embed - n_lm_head

    return {"total": f"{n_total:,}", "backbone": f"{n_backbone:,}"}


class GLUExpertWeightFuser:
    """
    A utility class to fuse and split the weights of GLU experts in a
    Mixture-of-Experts (MoE) model.

    Its main function is to fuse the `up_proj` and `gate_proj` weights of each expert
    into a single tensor, and then stack the weights of all experts to form a
    format suitable for optimized computations like `grouped_gemm`. It also provides
    the reverse operation (splitting).
    """

    def __init__(
        self, key_mappings: dict[str, str], fused_key_mappings: dict[str, str]
    ):
        """
        Initializes the Fuser.

        Args:
            key_mappings (dict[str, str]): The key name formats for the original,
                individual expert weights.
                e.g., {"up_proj": "model.layers.{}.block_sparse_moe.experts.{}.up_proj.weight", ...}
            fused_key_mappings (dict[str, str]): The key name formats for the fused weights.
                e.g., {"up_proj": "model.layers.{}.block_sparse_moe.fused_experts.up_proj.weight", ...}
        """
        # Store the key name templates for the original, separate expert weights
        self.up_proj_key = key_mappings["up_proj"]
        self.gate_proj_key = key_mappings["gate_proj"]
        self.down_proj_key = key_mappings["down_proj"]

        # Store the key name templates for the fused weights
        self.up_proj_weight_key = fused_key_mappings["up_proj"]
        self.down_proj_weight_key = fused_key_mappings["down_proj"]

    def fuse_experts(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_inds: List[int],
        num_experts: int,
    ) -> dict[str, torch.Tensor]:
        """
        Fuses the separate expert weights in a state_dict into stacked tensors.

        Args:
            state_dict (dict[str, torch.Tensor]): The model's state_dict.
            layer_inds (List[int]): A list of indices for the MoE layers to be fused.
            num_experts (int): The number of experts in each MoE layer.

        Returns:
            dict[str, torch.Tensor]: A new state_dict containing the fused weights.
        """
        fused_states = {}
        for i in layer_inds:
            up_proj_weights = []
            down_proj_weights = []
            for j in range(num_experts):
                # Generate the specific weight keys based on the templates and indices
                up_proj_key = self.up_proj_key.format(i, j)
                gate_proj_key = self.gate_proj_key.format(i, j)
                down_proj_key = self.down_proj_key.format(i, j)

                # Pop (get and remove) the weights from the state_dict
                up_proj = state_dict.pop(up_proj_key)
                gate_proj = state_dict.pop(gate_proj_key)

                # Core operation: Concatenate up_proj and gate_proj along dimension 0.
                # This allows computing the activations for both up and gate in a single matrix multiplication.
                up_proj = torch.cat([up_proj, gate_proj], dim=0)

                down_proj = state_dict.pop(down_proj_key)

                # Collect the weights of all experts for the current layer
                up_proj_weights.append(up_proj)
                down_proj_weights.append(down_proj)

            # Core operation: Stack the weights of all experts in this layer to form a "grouped" tensor.
            # This adds a new dimension for the expert index, changing the shape to [num_experts, ...].
            up_proj_weight = torch.stack(up_proj_weights)
            down_proj_weight = torch.stack(down_proj_weights)

            # Generate the key names for the fused weights
            up_proj_weight_key = self.up_proj_weight_key.format(i)
            down_proj_weight_key = self.down_proj_weight_key.format(i)

            # Store them in the new state_dict
            fused_states[up_proj_weight_key] = up_proj_weight
            fused_states[down_proj_weight_key] = down_proj_weight

        # Add the remaining (non-expert) weights from the original state_dict to the new dictionary
        fused_states.update(state_dict)
        return fused_states

    def split_experts(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_inds: List[int],
        num_experts: int,
    ) -> dict[str, torch.Tensor]:
        """
        Splits the fused expert weights back into separate, individual expert weights.
        This is the inverse operation of `fuse_experts`.

        Args:
            state_dict (dict[str, torch.Tensor]): The state_dict containing the fused weights.
            layer_inds (List[int]): A list of indices for the MoE layers to be split.
            num_experts (int): The number of experts in each MoE layer.

        Returns:
            dict[str, torch.Tensor]: A new state_dict containing the split weights.
        """
        outputs = {}
        for i in layer_inds:
            # Get the key names for the fused weights
            up_proj_weight_key = self.up_proj_weight_key.format(i)
            down_proj_weight_key = self.down_proj_weight_key.format(i)

            # Pop the fused weights from the state_dict
            up_proj_weight: torch.Tensor = state_dict.pop(up_proj_weight_key)
            down_proj_weight = state_dict.pop(down_proj_weight_key)

            # Core operation: Split the fused up/gate weights into two parts along dimension 1.
            up_proj_weight, gate_proj_weight = up_proj_weight.chunk(2, dim=1)

            # Ensure the resulting tensors are contiguous in memory, which can be necessary for subsequent operations.
            up_proj_weight = up_proj_weight.contiguous()
            gate_proj_weight = gate_proj_weight.contiguous()

            for j in range(num_experts):
                # Extract the weights for a single expert from the stacked tensor
                up_proj = up_proj_weight[j]
                gate_proj = gate_proj_weight[j]
                down_proj = down_proj_weight[j]

                # Generate the original expert weight keys
                up_proj_key = self.up_proj_key.format(i, j)
                gate_proj_key = self.gate_proj_key.format(i, j)
                down_proj_key = self.down_proj_key.format(i, j)

                # Store the split weights in the new state_dict
                outputs[up_proj_key] = up_proj
                outputs[gate_proj_key] = gate_proj
                outputs[down_proj_key] = down_proj

        # Add the remaining weights from the original state_dict to the new dictionary
        outputs.update(state_dict)
        return outputs
