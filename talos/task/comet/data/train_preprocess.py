#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/02 21:27:13
# Author: Shilei Liu
import math
import random
from copy import deepcopy
from typing import List

import torch

from talos.task.comet.data.chunk_plan import ChunkPlanner
from talos.task.comet.data.input_builder import InputBuilder
from talos.task.comet.utils import calc_static_chunk_sizes


class TrainTransform:
    ignore_id: int = -100

    def __init__(
        self,
        max_len: int,
        pad_id: int,
        placeholder_id: int,
        chunk_size: int,
        temp_beacon_id: int,
        temp_beacon_stride: int,
        temp_mem_budget: int,
        global_beacon_ids: List[int],
        planner: ChunkPlanner,
    ):
        self.max_len = max_len
        self.pad_id = pad_id
        self.ph_id = placeholder_id
        self.chunk_size = chunk_size
        assert max_len % chunk_size == 0, "max_len must be divisible by chunk_size"
        self.num_chunks = max_len // chunk_size
        self.temp_beacon_id = temp_beacon_id
        self.temp_beacon_stride = temp_beacon_stride
        self.temp_mem_budget = temp_mem_budget
        self.global_beacon_ids = global_beacon_ids
        self.planner = planner
        self.num_beacons = max(math.ceil(chunk_size / temp_beacon_stride), 0)
        self.max_compress_ratio = -1

    @classmethod
    def from_config(cls, config, max_len: int, planner: ChunkPlanner):
        return cls(
            max_len,
            config.pad_token_id,
            config.placeholder_id,
            config.chunk_size,
            config.temp_beacon_id,
            config.temp_beacon_stride,
            config.temp_mem_budget,
            config.global_beacon_ids,
            planner,
        )

    def enable_dynamic_compress_ratio(self, max_ratio: int):
        self.max_compress_ratio = max_ratio

    @staticmethod
    def sample_power_of_2(low: int, high: int) -> int:
        def is_power_of_two(n: int) -> bool:
            if n <= 0:
                return False
            return (n & (n - 1)) == 0

        assert low <= high
        assert is_power_of_two(low) and is_power_of_two(high)

        population = []
        current_val = low
        while current_val <= high:
            population.append(current_val)
            current_val *= 2

        num_items = len(population)
        weights = [(num_items - i) ** 2 for i in range(num_items)]

        return random.choices(population, weights=weights, k=1)[0]

    def compress_ratio(self):
        if self.max_compress_ratio != -1:
            low = self.temp_beacon_stride
            high = self.max_compress_ratio
            stride = self.sample_power_of_2(low, high)
        else:
            stride = self.temp_beacon_stride
        return stride

    def __call__(self, line):
        input_ids = line[0]
        # If labels are not provided, use a deepcopy of input_ids as the base.
        labels = line[1] if len(line[1]) > 0 else deepcopy(line[0])

        # Truncate the input to the maximum processable length.
        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]

        chunk_sizes = self.planner.plan(input_ids, self.max_len, self.chunk_size)
        builder = InputBuilder.init(chunk_sizes, input_ids, labels)

        max_chunk_sizes = calc_static_chunk_sizes(
            self.max_len,
            self.chunk_size,
            len(self.global_beacon_ids),
            self.temp_beacon_stride,
            self.temp_mem_budget,
        )

        builder.pad_num_chunks(self.num_chunks)
        builder.inject_temp_memory_tokens(
            self.temp_beacon_id,
            self.compress_ratio(),
            self.temp_mem_budget,
            self.ph_id,
        )
        builder.inject_global_memory_tokens(self.global_beacon_ids, self.ph_id)
        builder.pad_temp_memory_beacons(
            self.num_beacons, self.temp_mem_budget, self.pad_id
        )
        builder.pad_chunk_sizes(max_chunk_sizes, self.pad_id)
        builder.init_position_ids()
        seq = builder.fetch()
        return seq


class TrainBatchCollator:
    def __call__(self, examples):
        if examples is None or examples[0] is None:
            return examples
        chunk_sizes = examples[0]["chunk_sizes"]
        input_ids = torch.stack([e["input_ids"] for e in examples])
        labels = torch.stack([e["labels"] for e in examples])
        attention_mask = torch.stack([e["attention_mask"] for e in examples])
        position_ids = torch.stack([e["position_ids"] for e in examples])

        t_mem_inds_list = [e["temp_mem_inds"] for e in examples]
        t_beacon_inds_list = [e["temp_beacon_inds"] for e in examples]
        t_mem_select_inds_list = [e["temp_mem_select_inds"] for e in examples]
        g_mem_inds_list = [e["global_mem_inds"] for e in examples]
        g_beacon_inds_list = [e["global_beacon_inds"] for e in examples]

        num_chunks = len(chunk_sizes)

        batch_g_mem_inds = []
        batch_g_beacon_inds = []
        batch_t_mem_inds = []
        batch_t_beacon_inds = []
        batch_t_select_inds = []

        for i in range(num_chunks):
            ith_g_mem_inds = torch.stack([t[i] for t in g_mem_inds_list])
            ith_g_beacon_inds = torch.stack([t[i] for t in g_beacon_inds_list])
            ith_t_mem_inds = torch.stack([t[i] for t in t_mem_inds_list])
            ith_t_beacon_inds = torch.stack([t[i] for t in t_beacon_inds_list])
            ith_t_select_inds = torch.stack([t[i] for t in t_mem_select_inds_list])

            batch_g_mem_inds.append(ith_g_mem_inds)
            batch_g_beacon_inds.append(ith_g_beacon_inds)
            batch_t_mem_inds.append(ith_t_mem_inds)
            batch_t_beacon_inds.append(ith_t_beacon_inds)
            batch_t_select_inds.append(ith_t_select_inds)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "chunk_sizes": chunk_sizes,
            "global_mem_inds": batch_g_mem_inds,
            "global_beacon_inds": batch_g_beacon_inds,
            "temp_mem_inds": batch_t_mem_inds,
            "temp_beacon_inds": batch_t_beacon_inds,
            "temp_mem_select_inds": batch_t_select_inds,
        }
