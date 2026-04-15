#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2023/12/31 23:59:05
# Author: Shilei Liu
import gc
import json
import os
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)

from talos.dist import is_world_first_process
from talos.io.loader import CkptLoader, load_in_cpu
from talos.io.paths import (
    OPTIMIZER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    issue_warnings_after_load,
)
from talos.utils import TrainState, get_logger


__all__ = [
    "HFCkptLoader",
    "low_mem_load_model",
    "load_sharded_state_dict",
]

logger = get_logger()


def load_sharded_state_dict(folder, prefer_safe=True):
    weights_file = os.path.join(folder, WEIGHTS_NAME)
    safe_weights_file = os.path.join(folder, SAFE_WEIGHTS_NAME)

    # Load direct.
    if os.path.isfile(weights_file):
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        return state_dict
    elif os.path.isfile(safe_weights_file):
        state_dict = safe_load_file(safe_weights_file)
        return state_dict

    # Load the index.
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME)
            if is_safetensors_available()
            else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(
            f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}."
        )

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # load safe since we have no other choice

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    if load_safe:
        loader = safe_load_file
    else:
        loader = partial(torch.load, map_location="cpu", weights_only=True)

    full_state_dict = {}

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        full_state_dict.update(state_dict)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    return full_state_dict


def low_mem_load_model(
    model: nn.Module,
    dir: Optional[str] = None,
    state_dict: Optional[dict[str, Any]] = None,
):
    if dir is not None and is_world_first_process():
        assert state_dict is None
        state_dict = load_sharded_state_dict(dir)
    result = set_model_state_dict(
        model=model,
        model_state_dict=state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )
    issue_warnings_after_load(result)
    torch.cuda.empty_cache()
    return result


class HFCkptLoader(CkptLoader):
    def _load_model(self, dir: str, model: nn.Module):
        logger.info("Loading parameters...")
        low_mem_load_model(model, dir)

    def _load_optim(self, dir: str, model: nn.Module, optimizer: Optimizer):
        if optimizer is None or self.args.no_load_optim:
            return
        logger.info("Loading optimizer...")
        if is_world_first_process():
            full_sd = load_in_cpu(os.path.join(dir, OPTIMIZER_NAME))
        else:
            full_sd = None
        set_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            optim_state_dict=full_sd,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        torch.cuda.empty_cache()

    def _load_scheduler(self, dir: str, scheduler: LRScheduler):
        if scheduler is not None and not self.args.no_load_lrs:
            sch_states = load_in_cpu(os.path.join(dir, SCHEDULER_NAME))
            scheduler.load_state_dict(sch_states)

    def _load_scaler(self, dir: str, scaler: ShardedGradScaler):
        if scaler is not None and not self.args.no_load_scaler:
            scaler_states = load_in_cpu(os.path.join(dir, SCALER_NAME))
            scaler.load_state_dict(scaler_states)

    def load(
        self,
        dir: str,
        state: TrainState,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: ShardedGradScaler,
        from_remote: bool,
    ):
        if from_remote:
            buf = self._copy_ckpts_to_cache(dir)
        else:
            buf = dir
        self._load_model(buf, model)
        self._load_optim(buf, model, optimizer)
        self._load_scheduler(buf, scheduler)
        self._load_scaler(buf, scaler)
        self._load_rng_state(buf)
        self._load_train_state(buf, state)
