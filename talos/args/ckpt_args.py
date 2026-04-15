#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/09 23:35:49
# Author: Shilei Liu
from dataclasses import dataclass, field
from typing import Optional


__all__ = ["CkptArguments"]


@dataclass
class CkptArguments:
    save: str = field(
        default="/tmp/save",
        metadata={"help": "Save path."},
    )
    load: str = field(
        default="/tmp/load",
        metadata={"help": "Load path."},
    )
    remote_save: Optional[str] = field(
        default=None,
        metadata={"help": "Remote save path."},
    )
    remote_load: Optional[str] = field(
        default=None,
        metadata={"help": "Remote load path."},
    )
    no_load_rng: bool = field(default=False)
    no_load_state: bool = field(default=False)
    no_load_optim: bool = field(default=False)
    no_load_lrs: bool = field(default=False)
    no_load_scaler: bool = field(default=False)

    def load_path(self) -> str:
        return self.remote_load if self.remote_load is not None else self.load
