#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/02 21:27:13
# Author: Shilei Liu
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class InputBuilder:
    num_chunks: int
    chunk_sizes: List[int]
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    is_pad_chunk: List[bool]
    labels: Optional[List[List[int]]] = None
    global_mem_inds: Optional[List[List[int]]] = field(default_factory=list)
    global_beacon_inds: Optional[List[List[int]]] = field(default_factory=list)
    temp_mem_inds: Optional[List[List[int]]] = field(default_factory=list)
    temp_beacon_inds: Optional[List[List[int]]] = field(default_factory=list)
    temp_mem_select_inds: Optional[List[List[int]]] = field(default_factory=list)
    position_ids: Optional[List[List[int]]] = None
    ignore_id: int = -100

    @classmethod
    def init(
        cls,
        chunk_sizes: List[int],
        input_ids: List[int],
        labels: Optional[List[int]] = None,
    ) -> "InputBuilder":
        if labels is None:
            labels = input_ids[:]
        # Shift labels by removing first token and appending ignore_id.
        labels = labels[1:] + [cls.ignore_id]
        cur_pos = 0
        input_ids_list = []
        labels_list = []
        attention_mask = []
        is_pad_chunk = []

        for s in chunk_sizes:
            input_ids_list.append(input_ids[cur_pos : cur_pos + s])
            labels_list.append(labels[cur_pos : cur_pos + s])
            attention_mask.append([1] * s)
            is_pad_chunk.append(False)
            cur_pos += s

        return cls(
            num_chunks=len(chunk_sizes),
            chunk_sizes=chunk_sizes,
            input_ids=input_ids_list,
            attention_mask=attention_mask,
            labels=labels_list,
            is_pad_chunk=is_pad_chunk,
        )

    @staticmethod
    def vector_add_(vector: List[int], val: int):
        """
        Performs an in-place addition of a scalar value to all elements of a list of integers.
        Used for adjusting indices after left-padding.
        """
        for i in range(len(vector)):
            vector[i] += val

    def pad_num_chunks(self, max_size: int):
        assert max_size >= self.num_chunks
        diff = max_size - self.num_chunks
        self.num_chunks += diff
        zero_pad = [0 for i in range(diff)]
        empty_pad = [[] for i in range(diff)]
        bool_pad = [True for i in range(diff)]

        self.chunk_sizes = self.chunk_sizes + zero_pad
        self.input_ids = self.input_ids + empty_pad
        self.labels = self.labels + empty_pad
        self.attention_mask = self.attention_mask + empty_pad
        self.is_pad_chunk = self.is_pad_chunk + bool_pad

    def inject_temp_memory_tokens(
        self,
        temp_mem_beacon_id: int,
        temp_mem_beacon_stride: int,
        temp_mem_budget: int,
        ph_id: int,
        is_generating: bool = False,
    ):
        if temp_mem_beacon_stride == -1:
            self.temp_beacon_inds = [[] for _ in range(self.num_chunks)]
            self.temp_mem_inds = [[] for _ in range(self.num_chunks)]
            self.temp_mem_select_inds = [[] for _ in range(self.num_chunks)]
            return
        temp_mem_beacon_inds = []
        temp_mem_inds = []
        temp_mem_select_inds = []
        prefix_size = 0
        for i in range(self.num_chunks):
            cur_temp_mem_beacon_inds = []
            cur_temp_mem_inds = list(range(prefix_size))
            cur_select_inds = []

            new_input_ids = [ph_id] * prefix_size
            new_attention_mask = [1] * prefix_size
            new_labels = [self.ignore_id] * prefix_size

            orig_seq_len = len(self.input_ids[i])

            for beg in range(0, orig_seq_len, temp_mem_beacon_stride):
                end = min(beg + temp_mem_beacon_stride, orig_seq_len)
                new_input_ids.extend(self.input_ids[i][beg:end])
                new_attention_mask.extend(self.attention_mask[i][beg:end])
                new_labels.extend(self.labels[i][beg:end])

                if beg + temp_mem_beacon_stride >= orig_seq_len and is_generating:
                    break

                cur_temp_mem_beacon_inds.append(len(new_input_ids))
                new_input_ids.append(temp_mem_beacon_id)
                new_attention_mask.append(1)
                new_labels.append(self.ignore_id)

            temp_mem_beacon_inds.append(cur_temp_mem_beacon_inds)
            temp_mem_inds.append(cur_temp_mem_inds)
            self.input_ids[i] = new_input_ids
            self.attention_mask[i] = new_attention_mask
            self.labels[i] = new_labels
            self.chunk_sizes[i] = len(new_input_ids)

            prefix_size += len(cur_temp_mem_beacon_inds)
            cur_select_inds = list(
                range(max(0, prefix_size - temp_mem_budget), prefix_size)
            )
            temp_mem_select_inds.append(cur_select_inds)

            prefix_size = min(prefix_size, temp_mem_budget)

        self.temp_beacon_inds = temp_mem_beacon_inds
        self.temp_mem_inds = temp_mem_inds
        self.temp_mem_select_inds = temp_mem_select_inds

    def inject_global_memory_tokens(
        self, global_mem_beacon_ids: List[int], ph_id: int, is_generating: bool = False
    ):
        global_mem_beacon_inds = []
        global_mem_inds = []
        pool_size = len(global_mem_beacon_ids)
        for i in range(self.num_chunks):
            cur_global_mem_beacon_inds = []
            cur_global_mem_inds = []

            new_input_ids = []
            new_attention_mask = []
            new_labels = []
            if i > 0:
                new_input_ids.extend([ph_id] * pool_size)
                new_attention_mask.extend([1] * pool_size)
                new_labels.extend([self.ignore_id] * pool_size)
                if len(self.temp_mem_inds) > 0:
                    self.vector_add_(self.temp_mem_inds[i], pool_size)
                    self.vector_add_(self.temp_beacon_inds[i], pool_size)
                cur_global_mem_inds.extend(list(range(pool_size)))

            new_input_ids.extend(self.input_ids[i])
            new_attention_mask.extend(self.attention_mask[i])
            new_labels.extend(self.labels[i])

            if not (is_generating and i == self.num_chunks - 1):
                cur_len = len(new_input_ids)
                cur_global_mem_beacon_inds = list(range(cur_len, cur_len + pool_size))

                new_input_ids.extend(global_mem_beacon_ids)
                new_attention_mask.extend([1] * pool_size)
                new_labels.extend([self.ignore_id] * pool_size)

            global_mem_beacon_inds.append(cur_global_mem_beacon_inds)
            global_mem_inds.append(cur_global_mem_inds)

            self.input_ids[i] = new_input_ids
            self.attention_mask[i] = new_attention_mask
            self.labels[i] = new_labels
            self.chunk_sizes[i] = len(new_input_ids)

        self.global_beacon_inds = global_mem_beacon_inds
        self.global_mem_inds = global_mem_inds

    def init_position_ids(self):
        position_ids_list = []
        for attention_mask in self.attention_mask:
            position_ids = []
            idx = 0
            for m in attention_mask:
                position_ids.append(idx)
                idx += m
            position_ids_list.append(position_ids)
        self.position_ids = position_ids_list

    def pad_temp_memory_beacons(self, num_beacons: int, budget: int, pad_id: int):
        # 1. Record the number of real (non-pad) beacons generated by each chunk initially.
        if num_beacons == -1:
            return
        real_beacon_counts = [len(beacons) for beacons in self.temp_beacon_inds]

        new_select_inds = []
        # `kept_mems_info` stores (is_real, mem_id) tuples for the memory queue passed to the *next* chunk.
        kept_mems_info = []
        mem_id_counter = 0

        for i in range(self.num_chunks):
            # --- Part A: Pad the current chunk `i` ---

            # `mems_for_input` is the state of the queue from the previous step's merge. This is the input to chunk `i`.
            mems_for_input = kept_mems_info
            exp_num_mem = len(mems_for_input)

            # Pad memory placeholders (`temp_mem_inds`) if needed.
            seq_len = len(self.input_ids[i])
            cur_num_mem = len(self.temp_mem_inds[i])
            offset1 = max(0, exp_num_mem - cur_num_mem)

            # Pad beacon placeholders (`temp_beacon_inds`) to a fixed size `num_beacons`.
            cur_num_beacons = len(self.temp_beacon_inds[i])
            offset2 = num_beacons - cur_num_beacons
            offset = offset1 + offset2

            self.temp_mem_inds[i].extend(range(seq_len, seq_len + offset1))
            self.temp_beacon_inds[i].extend(range(seq_len + offset1, seq_len + offset))
            self.input_ids[i].extend([pad_id] * offset)
            self.labels[i].extend([self.ignore_id] * offset)
            self.attention_mask[i].extend([0] * offset)
            self.chunk_sizes[i] += offset

            # --- Part B: Calculate the merge order for chunk `i` and update state for chunk `i+1` ---

            # Identify new memories generated by this chunk, distinguishing real from pad.
            num_real = real_beacon_counts[i]
            num_pad = num_beacons - num_real

            mems_generated_by_current = []
            for _ in range(num_real):
                mems_generated_by_current.append((True, mem_id_counter))
                mem_id_counter += 1
            for _ in range(num_pad):
                mems_generated_by_current.append((False, mem_id_counter))
                mem_id_counter += 1

            # At the end of chunk `i`, merge its input memories with its newly generated memories.
            combined_mems = mems_for_input + mems_generated_by_current

            # Sort to determine eviction priority: real > pad, then newer > older.
            sorted_mems = sorted(
                combined_mems, key=lambda m: (m[0], m[1]), reverse=True
            )

            # Apply budget to get the memories that will be kept for the next chunk.
            mems_to_keep_for_next = sorted_mems[:budget]

            # Calculate the merge indices for this step, relative to `concat(mems_for_input, mems_generated_by_current)`.
            prev_mem_map = {
                mem_id: idx for idx, (is_real, mem_id) in enumerate(mems_for_input)
            }
            curr_mem_map = {
                mem_id: idx
                for idx, (is_real, mem_id) in enumerate(mems_generated_by_current)
            }

            current_merge_order = []
            for is_real, mem_id in mems_to_keep_for_next:
                if mem_id in prev_mem_map:
                    current_merge_order.append(prev_mem_map[mem_id])
                else:
                    idx_in_new = curr_mem_map[mem_id]
                    current_merge_order.append(len(mems_for_input) + idx_in_new)

            current_merge_order.sort()
            new_select_inds.append(current_merge_order)

            # Update the state for the next iteration. `kept_mems_info` must be sorted by ID to maintain order.
            kept_mems_info = sorted(mems_to_keep_for_next, key=lambda m: m[1])

        # Overwrite the old merge orders with the new priority-based ones.
        self.temp_mem_select_inds = new_select_inds

    def pad_chunk_sizes(self, chunk_sizes: List[int], pad_id: int):
        assert len(chunk_sizes) == len(self.chunk_sizes)
        for i in range(self.num_chunks):
            diff = chunk_sizes[i] - self.chunk_sizes[i]
            assert diff >= 0, f"[{i}] {chunk_sizes[i]} vs {self.chunk_sizes[i]}"
            source_pads = [pad_id] * diff
            target_pads = [self.ignore_id] * diff
            zeros_pads = [0] * diff
            self.input_ids[i] = self.input_ids[i] + source_pads
            self.labels[i] = self.labels[i] + target_pads
            self.attention_mask[i] = self.attention_mask[i] + zeros_pads
            self.chunk_sizes[i] += diff

    def fetch(self, training: bool = True):
        def _to_tensor_list(lst: List[List[int]]) -> List[torch.Tensor]:
            return [torch.tensor(t) for t in lst]

        def _to_tensor(lst: List[List[int]]) -> torch.tensor:
            return torch.cat(_to_tensor_list(lst))

        input_ids = _to_tensor(self.input_ids)
        attention_mask = _to_tensor(self.attention_mask)
        position_ids = _to_tensor(self.position_ids)

        temp_mem_inds = _to_tensor_list(self.temp_mem_inds)
        temp_beacon_inds = _to_tensor_list(self.temp_beacon_inds)
        temp_mem_select_inds = _to_tensor_list(self.temp_mem_select_inds)
        global_mem_inds = _to_tensor_list(self.global_mem_inds)
        global_beacon_inds = _to_tensor_list(self.global_beacon_inds)
        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "chunk_sizes": self.chunk_sizes,
            "temp_mem_inds": temp_mem_inds,
            "temp_beacon_inds": temp_beacon_inds,
            "temp_mem_select_inds": temp_mem_select_inds,
            "global_mem_inds": global_mem_inds,
            "global_beacon_inds": global_beacon_inds,
        }

        if training:
            labels = _to_tensor(self.labels)
            sample["labels"] = labels

        return sample
