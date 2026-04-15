#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/11 23:56:49
# Author: Shilei Liu
from typing import Any, List, Type

import torch.nn as nn
import torch.optim as optim

from talos.args import TrainingArguments


__all__ = [
    "get_parameter_names",
    "group_params_default",
    "get_optimizers",
]


def get_parameter_names(
    model: nn.Module, forbidden_layer_types: List[Type]
) -> List[str]:
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def group_params_default(
    model: nn.Module, weight_decay: float, forbidden_layer_types: List[Type]
):
    decay_param_names = get_parameter_names(model, forbidden_layer_types)
    decay_param_names = [
        name for name in decay_param_names if not name.endswith(".bias")
    ]
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n not in decay_param_names:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return grouped_parameters


def get_optimizers(args: TrainingArguments, params: List[dict[str, Any]]):
    optimizer = optim.AdamW(
        params,
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    return optimizer
