#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/10/07 18:10:53
# Author: Shilei Liu
from talos.data.base import FiniteStreamDataset


__all__ = ["DatasetPlaceholder"]


class DatasetPlaceholder(FiniteStreamDataset):
    def __init__(
        self,
        row_count_local: int,
        row_count_total: int,
        slice_id: int,
        slice_count: int,
    ):
        super().__init__(slice_id, slice_count)
        self.row_count_local = row_count_local
        self.row_count_total = row_count_total

    def reset_data_ptr(self, consumed_samples: int = 0):
        pass

    @property
    def num_samples_local(self) -> int:
        return self.row_count_local

    @property
    def num_samples_global(self) -> int:
        return self.row_count_total

    def build_data_stream(self, worker_id: int, num_workers: int):
        while True:
            yield None
