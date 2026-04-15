#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/10 20:39:02
# Author: Shilei Liu
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch

from talos.args.ckpt_args import CkptArguments
from talos.dist import get_world_size
from talos.utils import get_logger


__all__ = [
    "LoggerArguments",
    "ProfileArguments",
    "OptimArguments",
    "MixedPrecArguments",
    "TrainingArguments",
]

logger = get_logger()


@dataclass
class LoggerArguments:
    log_interval: int = field(default=10)
    timing_log_level: int = field(default=0)
    timing_log_option: str = field(default="minmax")
    barrier_with_l1_time: Optional[bool] = field(default=True)
    tracker_impl: str = field(
        default="none", metadata={"choices": ["none", "ml_tracker", "tensorboard"]}
    )
    tracker_proj: Optional[str] = field(default=None)
    tracker_runid: Optional[str] = field(default=None)
    tracker_log_dir: Optional[str] = field(default=None)


@dataclass
class ProfileArguments:
    do_profile: bool = field(default=False)
    trace_path: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.do_profile:
            assert self.trace_path is not None


@dataclass
class OptimArguments:
    scheduler: str = field(
        default="linear",
        metadata={"choices": ["linear", "cosine"]},
    )
    min_lr: float = field(
        default=0.0,
        metadata={"help": "Min learning rate."},
    )
    max_lr: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay."},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for Adam optimizer."},
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for AdamW optimizer."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for AdamW optimizer"},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )


@dataclass
class MixedPrecArguments:
    dtype: str = field(
        default="fp32",
        metadata={"choices": ["fp32", "bf16", "fp16"]},
    )

    def torch_dtype(self):
        mapper = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return mapper[self.dtype]

    def dtype_string(self):
        mapper = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }
        return mapper[self.dtype]


@dataclass
class TrainingArguments(
    LoggerArguments, ProfileArguments, OptimArguments, MixedPrecArguments, CkptArguments
):
    seed: int = field(default=42, metadata={"help": "Random seed."})
    save_interval: int = field(default=10000)
    fsdp_checkpoint_activations: Optional[bool] = field(default=False)
    micro_train_batch_size: int = field(default=4)
    global_train_batch_size: int = field(default=-1)
    micro_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    max_grad_norm: float = field(default=1.5)
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform."},
    )
    num_train_samples: int = field(
        default=-1,
        metadata={"help": "Num samples to train."},
    )
    num_epochs: int = field(default=1, metadata={"help": "Num train epochs."})
    num_workers: int = field(
        default=4,
        metadata={"help": "Num workers of dataloader."},
    )
    tp_size: int = field(default=1)
    pp_size: int = field(default=1)
    cp_size: int = field(default=1)

    def adjust_global_train_batch_size(self):
        world_size = get_world_size()
        mbs = self.micro_train_batch_size
        ga = self.gradient_accumulation_steps

        tp_size, pp_size, cp_size = self.tp_size, self.pp_size, self.cp_size
        p_size = tp_size * pp_size * cp_size
        assert world_size >= p_size
        assert world_size % p_size == 0
        gbs_pred = mbs * world_size * ga // p_size
        gbs_gold = self.global_train_batch_size

        if gbs_gold == -1:
            self.global_train_batch_size = gbs_pred
            logger.info(f"Set global_train_batch_size to {gbs_pred}")
        elif gbs_gold != gbs_pred:
            assert gbs_gold % gbs_pred == 0
            ga = gbs_gold // gbs_pred
            self.gradient_accumulation_steps = ga
            logger.info(f"Set gradient_accumulation_steps to {ga}")

    def __post_init__(self):
        super().__post_init__()
        self.adjust_global_train_batch_size()

        gbs = self.global_train_batch_size

        if self.num_train_samples != -1:
            assert self.max_steps == -1
            self.max_steps = self.num_train_samples // gbs
            logger.info(
                f"Set max_steps to {self.max_steps} according to num_train_samples"
            )

        assert self.max_lr >= self.min_lr

        if self.tracker_impl != "none":
            assert self.tracker_proj is not None or self.tracker_log_dir is not None

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        return args
