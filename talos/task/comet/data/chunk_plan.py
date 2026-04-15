#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/12/20 18:34:29
# Author: Shilei Liu
import random
from typing import List

from talos.task.comet.data.chunk_strategy import (
    DynamicNumRandomSizeStrategy,
    FixedSizeStrategy,
    RightAlignedFixedSizeStrategy,
    SemanticBoundaryStrategy,
    UniformSizeStrategy,
)


class ChunkPlanner:
    def plan(self, input_ids: List[int], max_len: int, max_chunk_size: int):
        pass


class DefaultChunkPlanner(ChunkPlanner):
    def __init__(self, boundary_token_ids: List[int]):
        self.boundary_token_ids = set(boundary_token_ids)

    def plan(
        self, input_ids: List[int], max_len: int, max_chunk_size: int
    ) -> List[int]:
        seq_len = len(input_ids)
        assert max_len >= 4
        assert max_len % max_chunk_size == 0
        max_num_chunks = max_len // max_chunk_size
        rnd = random.random()
        if seq_len < max_chunk_size:
            if rnd < 0.8:
                chunk_sizes = FixedSizeStrategy.build(seq_len, max_chunk_size)
            elif rnd < 0.9:
                chunk_sizes = RightAlignedFixedSizeStrategy.build(
                    seq_len, max_chunk_size
                )
            else:
                chunk_sizes = DynamicNumRandomSizeStrategy.build(
                    seq_len, max_num_chunks, 4, max_chunk_size
                )
        elif seq_len < 0.75 * max_len:
            if rnd < 0.6:
                chunk_sizes = FixedSizeStrategy.build(seq_len, max_chunk_size)
            elif rnd < 0.7:
                chunk_sizes = RightAlignedFixedSizeStrategy.build(
                    seq_len, max_chunk_size
                )
            elif rnd < 0.9:
                chunk_sizes = DynamicNumRandomSizeStrategy.build(
                    seq_len, max_num_chunks, 4, max_chunk_size
                )
            elif rnd < 0.95:
                chunk_sizes = UniformSizeStrategy.build(seq_len, max_num_chunks)
            else:
                chunk_sizes = SemanticBoundaryStrategy.build(
                    input_ids,
                    max_chunk_size,
                    4,
                    max_num_chunks,
                    self.boundary_token_ids,
                )
        else:
            if rnd < 0.6:
                chunk_sizes = FixedSizeStrategy.build(seq_len, max_chunk_size)
            elif rnd < 0.7:
                chunk_sizes = RightAlignedFixedSizeStrategy.build(
                    seq_len, max_chunk_size
                )
            elif rnd < 0.9:
                chunk_sizes = UniformSizeStrategy.build(seq_len, max_num_chunks)
            else:
                chunk_sizes = DynamicNumRandomSizeStrategy.build(
                    seq_len, max_num_chunks, 4, max_chunk_size
                )
        return chunk_sizes
