#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/10 16:52:03
# Author: Shilei Liu
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from talos.args import TrainingArguments
from talos.dist import get_local_rank
from talos.utils.timers import TimersPersistentState
from talos.utils.tools import set_seed
from talos.utils.tracker import TrackerPersistentState


__all__ = ["setup", "cleanup", "initialize"]


def setup():
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    kwargs = {"device_id": torch.device(f"cuda:{local_rank}")}
    if (sec := os.getenv("TORCH_DIST_TIMEOUT_SEC")) is not None:
        kwargs["timeout"] = timedelta(seconds=int(sec))
    dist.init_process_group("nccl", **kwargs)


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    tracker = TrackerPersistentState.get()
    if tracker is not None:
        tracker.close()


def initialize(args: TrainingArguments):
    # NOTE: To ensure compatibility with HDFS data sources,
    # the multiprocessing start method must be set to 'spawn'
    # when utilizing multi-process data loading (num_workers > 0).
    if os.getenv("MP_START_METHOD", "default").lower() == "spawn":
        mp.set_start_method("spawn", force=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    TimersPersistentState.init(args.timing_log_level, args.timing_log_option)
