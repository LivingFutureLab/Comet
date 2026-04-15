#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2023/12/31 23:42:26
# Author: Shilei Liu
import os
from typing import Optional

import torch.distributed as dist
from torch.nn.modules.module import _IncompatibleKeys

from talos.dist import get_global_rank
from talos.utils import get_logger


__all__ = [
    "PARAMETER_NAME",
    "OPTIMIZER_NAME",
    "SCALER_NAME",
    "SCHEDULER_NAME",
    "TRAIN_STATE_NAME",
    "get_checkpoint_dir",
    "get_rng_state_path",
    "get_sharded_params_dir",
    "get_sharded_optim_dir",
    "get_sharded_optim_path",
    "get_sharded_params_path",
]

logger = get_logger()


PARAMETER_NAME = "pytorch_model.bin"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
SCHEDULER_NAME = "scheduler.pt"
TRAIN_STATE_NAME = "train_state.json"


def get_checkpoint_dir(save_dir: str, global_step: int) -> str:
    assert save_dir is not None and save_dir != "", "`save_dir` should not be empty."
    sub_dir = f"checkpoint-{global_step}"
    full_dir = os.path.join(save_dir, sub_dir)
    return full_dir


def get_rng_state_path(dir: str, rank: Optional[int] = None):
    if rank is None:
        rank = get_global_rank()
    path = os.path.join(dir, "rng_states", f"rng_state_{rank}.bin")
    return path


def get_sharded_params_dir(dir: str):
    return os.path.join(dir, "sharded_model")


def get_sharded_optim_dir(dir: str):
    return os.path.join(dir, "sharded_optim")


def get_sharded_optim_path(dir: str, rank: Optional[int] = None):
    if rank is None:
        rank = dist.get_rank()
    return os.path.join(dir, "sharded_optim", f"optimizer_{rank}.bin")


def get_sharded_params_path(dir: str, rank: Optional[int] = None):
    if rank is None:
        rank = dist.get_rank()
    return os.path.join(dir, "sharded_model", f"pytorch_model_{rank}.bin")


def issue_warnings_after_load(load_result: "_IncompatibleKeys"):
    if len(load_result.missing_keys) != 0:
        logger.warning(
            f"Missing keys when loading model params: {load_result.missing_keys}."
        )
    if len(load_result.unexpected_keys) != 0:
        logger.warning(
            f"Unexpected keys when loading model params: {load_result.unexpected_keys}."
        )
