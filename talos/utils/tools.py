#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/10 23:08:46
# Author: Shilei Liu
import datetime
import gc
import random

import numpy as np
import psutil
import torch

from talos.utils.logger import get_logger


logger = get_logger()

__all__ = [
    "set_seed",
    "cleanup_before_training",
    "format_time",
    "tprint",
    "get_num_parameters",
]


def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy` and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_before_training() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vm = psutil.virtual_memory()
    total_ram = vm.total / (1024**3)
    used_ram = vm.used / (1024**3)
    device_props = torch.cuda.get_device_properties()
    total_gpu_mem = device_props.total_memory / (1024**3)
    allocated_gpu_mem = torch.cuda.memory_allocated() / (1024**3)
    logger.info(
        f"Memory cleanup completed. "
        f"System RAM: {total_ram:.1f} GB (used: {used_ram:.1f} GB) | "
        f"GPU memory: {total_gpu_mem:.1f} GB (allocated: {allocated_gpu_mem:.1f} GB)"
    )


def format_time(s: int) -> str:
    """Convert seconds to h:m:s"""
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}:{int(m)}:{int(s)}"


def tprint(*args, **kwargs):
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template = "[{}]"
    print(template.format(cur_time), *args, **kwargs)


def get_num_parameters(model: torch.nn.Module) -> dict[str, str]:
    n_params = sum(p.numel() for p in model.parameters())
    n_tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {"total": f"{n_params:,}", "trainable": f"{n_tr_params:,}"}
    return info
