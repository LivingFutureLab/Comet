#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2023/10/06 11:48:18
# Author: Shilei Liu
from typing import Union


def calc_slice_range(
    row_count: int, worker_id: int, num_workers: int, baseline: int = 0
):
    # div-mod split, each slice data count max diff 1
    size = int(row_count / num_workers)
    split_point = row_count % num_workers
    if worker_id < split_point:
        start = worker_id * (size + 1) + baseline
        end = start + (size + 1)
    else:
        start = split_point * (size + 1) + (worker_id - split_point) * size + baseline
        end = start + size
    return start, end


def cala_local_worker_resume_offset(offset: int, worker_id: int, num_workers: int):
    start, end = calc_slice_range(offset, worker_id, num_workers)
    return end - start


def drop_redundant_data(num_samples_global, num_samples_local, slice_count):
    """Make the number of samples consistent on each slice."""
    num_samples_local_clip = num_samples_global // slice_count
    num_samples_global_clip = num_samples_local_clip * slice_count
    num_clip_samples_global = num_samples_global - num_samples_global_clip
    num_clip_samples_local = num_samples_local - num_samples_local_clip
    return num_clip_samples_global, num_clip_samples_local


def extract_str(key: Union[str, bytes, int, float], default: str = ""):
    """
    `common-io` package will return bytpes-format string in PAI platform
    and str-format string in local.
    """
    if key is None:
        return default
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    elif isinstance(key, (int, float, str)):
        key = key
    else:
        raise ValueError("only support str, bytes, int and float.")
    return key
