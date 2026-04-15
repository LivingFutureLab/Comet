#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2023/12/10 19:09:32
# Author: Shilei Liu
import math
from typing import Iterator, Optional, TypeVar

import torch
from torch.utils.data import Dataset, DistributedSampler


__all__ = ["DistDataSampler"]

T_co = TypeVar("T_co", covariant=True)


class DistDataSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, 0, drop_last)
        self._start = 0

    def reset_data_ptr(self, consumed_samples: int):
        assert self._start % self.num_replicas == 0
        self._start = consumed_samples

    def __len__(self) -> int:
        return self.num_samples - self._start // self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self._start :]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples - self._start // self.num_replicas
        self._start = 0

        return iter(indices)
