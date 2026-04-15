#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/27 01:38:30
# Author: Shilei Liu
from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist


__all__ = ["ParallelState"]


@dataclass
class ParallelState:
    group: dist.ProcessGroup
    rank: Optional[int] = None
    world_size: Optional[int] = None
    is_first_rank: Optional[bool] = None
    is_last_rank: Optional[bool] = None

    def __post_init__(self):
        if self.rank is None:
            self.rank = dist.get_rank(self.group)

        if self.world_size is None:
            self.world_size = dist.get_world_size(self.group)

        if self.is_first_rank is None:
            self.is_first_rank = self.rank == 0

        if self.is_last_rank is None:
            self.is_last_rank = self.rank == self.world_size - 1
