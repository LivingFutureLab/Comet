#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/11 21:05:46
# Author: Shilei Liu
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy


__all__ = ["get_mp_policy"]


def get_mp_policy(dtype: str):
    if dtype == "fp32":
        param_dtype = torch.float32
        reduce_dtype = torch.float32
        output_dtype = torch.float32
    elif dtype == "fp16":
        param_dtype = torch.float16
        reduce_dtype = torch.float16
        output_dtype = torch.float16
    elif dtype == "bf16":
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32
        output_dtype = torch.bfloat16
    else:
        raise ValueError(f"Does not support dtype `{dtype}`")

    policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )

    return policy
