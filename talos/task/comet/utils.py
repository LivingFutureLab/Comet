#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/12/20 20:52:02
# Author: Shilei Liu
# Author: Shilei Liu
import math
from typing import List, Union

import torch
from torch import Tensor


__all__ = ["CometBatch", "to_device", "calc_static_chunk_sizes"]


CometBatch = dict[str, Union[Tensor, List[Tensor], List[int]]]


def to_device(batch: CometBatch, device: torch.device) -> CometBatch:
    results = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            v = v.to(device=device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], Tensor):
            v = [t.to(device=device) for t in v]
        results[k] = v
    return results


def calc_static_chunk_sizes(
    seq_length: int,
    chunk_size: int,
    global_mem_size: int,
    temp_beacon_stride: int,
    tem_mem_budget: int,
):
    assert seq_length % chunk_size == 0
    num_chunks = seq_length // chunk_size
    chunk_sizes = [chunk_size] * num_chunks

    if temp_beacon_stride > 0:
        num_beacons = math.ceil(chunk_size / temp_beacon_stride)
    else:
        num_beacons = 0
    num_temp_mem = 0
    for i in range(num_chunks):
        if i > 0:
            chunk_sizes[i] += num_temp_mem + global_mem_size
        chunk_sizes[i] += global_mem_size + num_beacons
        num_temp_mem = min(tem_mem_budget, num_temp_mem + num_beacons)
    return chunk_sizes
