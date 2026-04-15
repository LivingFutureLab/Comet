#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/09 22:54:49
# Author: Shilei Liu
from dataclasses import dataclass, field
from typing import Optional


__all__ = ["DataArguments"]


@dataclass
class DataArguments:
    tables: Optional[str] = field(
        default=None,
        metadata={"help": "Input table names."},
    )
    outputs: Optional[str] = field(
        default=None,
        metadata={"help": "Output table names."},
    )
    data_source: str = field(
        default="odps",
        metadata={
            "help": "Data source.",
            "choices": ["odps", "chunk", "tunnel_odps", "ailake", "custom"],
        },
    )
