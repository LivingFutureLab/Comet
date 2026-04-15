#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2023/12/31 23:59:05
# Author: Shilei Liu
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from talos.args import CkptArguments
from talos.dist import (
    get_global_ranks_on_local_node,
    get_local_rank,
    get_world_size,
    is_local_first_process,
)
from talos.io.cloud import pull
from talos.io.paths import (
    OPTIMIZER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    TRAIN_STATE_NAME,
    get_rng_state_path,
)
from talos.utils import TrainState, get_logger


__all__ = [
    "CkptLoader",
    "set_rng_state",
    "load_in_cpu",
]

logger = get_logger()


def set_rng_state(rng_state):
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.random.set_rng_state(rng_state["cpu"])
    if torch.cuda.is_available():
        if get_local_rank() == -1:
            torch.cuda.random.set_rng_state_all(rng_state["cuda"])
        else:
            torch.cuda.random.set_rng_state(rng_state["cuda"])


def load_in_cpu(path: str, weights_only: bool = False):
    with open(path, "rb") as f:
        states = torch.load(f, map_location="cpu", weights_only=weights_only)
    return states


class CkptLoader(object):
    def __init__(self, args: CkptArguments) -> None:
        self.args = args

    def _load_rng_state(self, dir: str):
        if not self.args.no_load_rng:
            full_path = get_rng_state_path(dir)
            rng_states = load_in_cpu(full_path)
            set_rng_state(rng_states)

    def _load_train_state(self, dir: str, state: TrainState):
        if not self.args.no_load_state:
            full_path = os.path.join(dir, TRAIN_STATE_NAME)
            with open(full_path, "r") as f:
                sd = json.load(f)
            state.load_state_dict(sd)

    def _copy_ckpts_to_cache(self, dir: str):
        ignore_patterns = []

        if self.args.no_load_optim:
            ignore_patterns.append(f"*{OPTIMIZER_NAME}")

        if self.args.no_load_lrs:
            ignore_patterns.append(f"*{SCHEDULER_NAME}")

        if self.args.no_load_rng:
            ignore_patterns.append("*/rng_states/*")
        else:
            world_size = get_world_size()
            local_ranks = get_global_ranks_on_local_node()
            minr, maxr = local_ranks[0], local_ranks[-1]
            for i in range(world_size):
                if i < minr or i > maxr:
                    ignore_patterns.append(get_rng_state_path("*", i))
            ignore_patterns

        if self.args.no_load_scaler:
            ignore_patterns.append(f"*{SCALER_NAME}")

        if self.args.no_load_state:
            ignore_patterns.append(f"*{TRAIN_STATE_NAME}")

        cache_dir = "/tmp/load_cache"

        logger.info("Downloading checkpoints from cloud...")
        if is_local_first_process():
            pull(dir, local_dir=cache_dir, ignore_patterns=ignore_patterns)
        dist.barrier()
        return cache_dir
