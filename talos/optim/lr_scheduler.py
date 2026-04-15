#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/11 23:53:26
# Author: Shilei Liu
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


__all__ = [
    "CosineAnnealingLRWithWarmup",
    "LinearAnnealingLRWithWarmup",
    "get_lr_scheduler",
]


class CosineAnnealingLRWithWarmup(LambdaLR):
    def __init__(
        self, optimizer, train_steps, max_lr, min_lr, warmup_steps=0, last_epoch=-1
    ):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            cur_steps = float(current_step - warmup_steps)
            total_steps = float(max(1, train_steps - warmup_steps))
            return (
                (max_lr + min_lr) / 2
                + (max_lr - min_lr) / 2 * math.cos(cur_steps / total_steps * math.pi)
            ) / max_lr

        super().__init__(optimizer, lr_lambda, last_epoch)


class LinearAnnealingLRWithWarmup(LambdaLR):
    def __init__(
        self, optimizer, train_steps, max_lr, min_lr, warmup_steps=0, last_epoch=-1
    ):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            cur_steps = float(current_step - warmup_steps)
            total_steps = float(max(1, train_steps - warmup_steps))
            step_size = (max_lr - min_lr) / total_steps
            cur_lr = max_lr - cur_steps * step_size
            return cur_lr / max_lr

        super().__init__(optimizer, lr_lambda, last_epoch)


def get_lr_scheduler(
    optimizer: Optimizer,
    impl: str,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
):
    if impl == "cosine":
        scheduler = CosineAnnealingLRWithWarmup(
            optimizer, max_steps, max_lr, min_lr, warmup_steps
        )
    elif impl == "linear":
        scheduler = LinearAnnealingLRWithWarmup(
            optimizer, max_steps, max_lr, min_lr, warmup_steps
        )
    else:
        raise KeyError(f"Does not support {impl} scheduler")
    return scheduler
