#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/10 15:09:50
# Author: Shilei Liu
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from talos.dist import is_world_first_process
from talos.io.paths import (
    OPTIMIZER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
)
from talos.io.saver import CkptSaver, torch_save
from talos.utils import TrainState, get_logger


logger = get_logger()


__all__ = [
    "HFCkptSaver",
]


class HFCkptSaver(CkptSaver):
    def _save_tokenizer(self, dir: str, tokenizer: "PreTrainedTokenizerBase"):
        if tokenizer is not None and is_world_first_process():
            tokenizer.save_pretrained(dir)

    def _save_model(self, dir: str, model: "PreTrainedModel"):
        logger.info("Saving parameters...")
        state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if is_world_first_process():
            model.save_pretrained(dir, state_dict=state_dict)
        dist.barrier()
        torch.cuda.empty_cache()

    def _save_optim(self, dir: str, model: nn.Module, optimizer: Optimizer):
        if optimizer is None:
            return
        logger.info("Saving optimizer...")
        optim_state_dict = get_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if is_world_first_process():
            torch_save(optim_state_dict, os.path.join(dir, OPTIMIZER_NAME))
        dist.barrier()
        torch.cuda.empty_cache()

    def _save_scheduler(self, dir: str, scheduler: LRScheduler):
        if is_world_first_process() and scheduler is not None:
            torch_save(scheduler.state_dict(), os.path.join(dir, SCHEDULER_NAME))

    def _save_scaler(self, dir: str, scaler: ShardedGradScaler):
        if is_world_first_process() and scaler is not None:
            torch_save(scaler.state_dict(), os.path.join(dir, SCALER_NAME))

    def save(
        self,
        dir: str,
        ckpt_id: str,
        state: TrainState,
        tokenizer: "PreTrainedTokenizerBase",
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: ShardedGradScaler,
        to_remote: bool,
    ):
        buf = self._pre_make_dirs(dir, ckpt_id, to_remote)
        logger.info(f"Saving checkpoint to {buf} ...")
        self._save_tokenizer(buf, tokenizer)
        self._save_model(buf, model)
        self._save_optim(buf, model, optimizer)
        self._save_scheduler(buf, scheduler)
        self._save_scaler(buf, scaler)
        self._save_rng_state(buf)
        self._save_train_state(buf, state)
        if dist.is_initialized():
            dist.barrier()
        if to_remote:
            self._copy_buffer_to_dst(buf, dir, ckpt_id)
