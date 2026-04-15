#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2023/10/28 16:54:59
# Author: Shilei Liu
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Union


__all__ = ["TrainState", "PersistentTrainState", "get_state"]

BaseType = Union[str, float, int, bool]


@dataclass
class TrainState:
    global_step: int = 0
    epoch: int = 0
    max_steps: int = 0
    log_history: List[dict[str, BaseType]] = field(default_factory=lambda: [])

    def to_dict(self):
        dict_obj = asdict(self)
        dict_obj.pop("log_history")
        return dict_obj

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]):
        return cls(**state_dict)

    def load_state_dict(self, state_dict: dict[str, Any]):
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)


class PersistentTrainState:
    state: Optional[TrainState] = None

    @classmethod
    def init(cls):
        state = TrainState()
        PersistentTrainState.state = state

    @classmethod
    def set(cls, state: TrainState):
        cls.state = state

    @classmethod
    def get(cls):
        return cls.state

    def __new__(cls):
        raise TypeError("Namespace class, non-instantiable")


def get_state() -> TrainState:
    return PersistentTrainState.get()
