#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/08/29 11:09:12
# Author: Shilei Liu
import torch.distributed as dist

__all__ = ["ProcessGroupPersistentState", "get_process_group"]


class ProcessGroupPersistentState:
    groups: dict[str, dist.ProcessGroup] = {}

    @classmethod
    def set(cls, name, group):
        cls.groups[name] = group

    @classmethod
    def get(cls, name):
        return cls.groups[name]

    @classmethod
    def keys(cls):
        return cls.groups.keys()

    def __new__(cls):
        raise TypeError("Namespace class, non-instantiable")


def get_process_group(name: str) -> dist.ProcessGroup:
    return ProcessGroupPersistentState.get(name)
