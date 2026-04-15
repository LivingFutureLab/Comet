#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/10 20:45:15
# Author: Shilei Liu
from typing import Any, Callable, Iterator, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from talos.args import TrainingArguments
from talos.data.base import FiniteStreamDataset
from talos.data.samplers import DistDataSampler
from talos.io.ckpt import find_latest_ckpt, is_leaf_ckpt_dir
from talos.io.mos import get_latest_uri, is_leaf_uri, is_mos_uri
from talos.optim import (
    get_lr_scheduler,
    group_params_default,
)
from talos.train.callback import CallbackHandler, TrainerCallback, default_callbacks
from talos.utils import TrainState, cleanup_before_training, get_logger, get_timers


__all__ = ["Trainer"]


logger = get_logger()


class Trainer:
    def __init__(
        self,
        args: TrainingArguments,
        dataset: FiniteStreamDataset,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        collate_fn: Optional[Callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        self.args = args
        self.dataset = dataset
        if isinstance(dataset, FiniteStreamDataset):
            self.sampler = None
            if args.num_epochs > 1:
                raise NotImplementedError("Stream dataset can only train one epoch.")
        else:
            self.sampler = DistDataSampler(dataset, drop_last=True)
        self.dataloader = self.init_dataloader(dataset, collate_fn)
        self.max_steps, self.steps_per_epoch = self.set_train_steps(self.dataloader)
        self.state = TrainState(max_steps=self.max_steps)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_lr_scheduler()
        self.scaler = self.create_scaler()
        self.resume_checkpoint()
        if callbacks is None:
            callbacks = default_callbacks(args)
        self.callback_handler = CallbackHandler(
            callbacks, model, tokenizer, self.optimizer, self.scheduler, self.scaler
        )
        logger.info("Trainer initialization completed.")

    def create_optimizer(self):
        args = self.args
        grouped_params = group_params_default(
            self.model, args.weight_decay, [nn.LayerNorm]
        )
        optimizer = torch.optim.AdamW(
            grouped_params,
            lr=args.max_lr,
            betas=(args.beta1, args.beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        return optimizer

    def create_lr_scheduler(self):
        args = self.args
        lrs = get_lr_scheduler(
            self.optimizer,
            args.scheduler,
            self.max_steps,
            args.max_lr,
            args.min_lr,
            args.warmup_steps,
        )
        return lrs

    def create_scaler(self):
        scaler = ShardedGradScaler(enabled=self.args.dtype == "fp16")
        return scaler

    def set_train_steps(self, dataloader: DataLoader):
        num_batches = len(dataloader)
        ga = self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = num_batches // ga
        max_steps = num_update_steps_per_epoch * self.args.num_epochs
        if self.args.max_steps > 0:
            max_steps = min(max_steps, self.args.max_steps)
        logger.info(f"Set max_steps to {max_steps}")
        return max_steps, num_update_steps_per_epoch

    def init_dataloader(self, dataset, collate_fn: Callable):
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.micro_train_batch_size,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return dataloader

    def compute_loss(self, model: nn.Module, batch: dict[str, Any]):
        outputs = model(**batch)
        if isinstance(outputs, tuple):
            loss = outputs[0]
        elif isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss
        return loss

    def micro_train_step(
        self,
        model: FSDPModule,
        dataiter: Iterator,
        scaler: ShardedGradScaler,
    ) -> torch.Tensor:
        args = self.args
        num_grad_accum = args.gradient_accumulation_steps
        self.callback_handler.on_substep_begin(args, self.state)
        timers = get_timers()

        timers("batch-generator", log_level=2).start()
        batch = next(dataiter)
        timers("batch-generator").stop()

        timers("forward-compute", log_level=2).start()
        loss: torch.Tensor = self.compute_loss(model, batch)
        loss = loss / num_grad_accum
        timers("forward-compute").stop()

        timers("backward-compute", log_level=2).start()
        scaler.scale(loss).backward()
        timers("backward-compute").stop()

        loss = loss.detach()

        self.callback_handler.on_substep_end(args, self.state)
        return loss

    def train_step(
        self,
        model: FSDPModule,
        dataiter: Iterator,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: ShardedGradScaler,
    ):
        args = self.args
        timers = get_timers()
        accu_loss = torch.zeros(1, dtype=torch.float32, device="cuda")
        ga = args.gradient_accumulation_steps

        timers("forward-backward", log_level=1).start(barrier=args.barrier_with_l1_time)
        suc = True
        for substep in range(ga):
            loss = self.micro_train_step(model, dataiter, scaler)
            if loss.isnan():
                suc = False
            else:
                accu_loss.add_(loss)
        timers("forward-backward").stop()

        timers("optimizer", log_level=1).start(barrier=args.barrier_with_l1_time)
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if grad_norm.isnan():
            suc = False
        if suc:
            scaler.step(optimizer)
            scheduler.step()
        else:
            accu_loss.zero_()
            grad_norm = torch.zeros_like(grad_norm)

        scaler.update()
        optimizer.zero_grad()
        timers("optimizer").stop()
        return accu_loss, grad_norm, suc

    @staticmethod
    def all_reduce_avg(tensor: torch.Tensor):
        reduced_loss = tensor.clone() / dist.get_world_size()
        dist.all_reduce(reduced_loss)
        return reduced_loss

    def get_throughput_metrics(self):
        timers = get_timers()
        state = self.state
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        gbs = self.args.global_train_batch_size
        global_step = state.global_step
        interval = self.args.log_interval
        consumed_samples = global_step * gbs
        time_per_step = elapsed_time / interval
        remaining_secs = (state.max_steps - global_step) * time_per_step
        qps = gbs / time_per_step
        return consumed_samples, qps, remaining_secs

    def log(
        self,
        accu_loss: torch.Tensor,
        accu_grad_norm: torch.Tensor,
        lr: float,
        grad_scale: float,
        num_skips: int,
        logs: Optional[dict[str, Any]] = None,
    ):
        interval = self.args.log_interval

        if logs is None:
            logs: dict[str, Any] = {}

        loss = self.all_reduce_avg(accu_loss).item() / interval
        grad_norm = accu_grad_norm.item() / interval
        consumed_samples, qps, remaining = self.get_throughput_metrics()
        logs["loss"] = loss
        logs["lr"] = lr
        logs["num_skips"] = num_skips
        logs["grad_scale"] = grad_scale
        logs["grad_norm"] = grad_norm
        logs["consumed_samples"] = consumed_samples
        logs["qps"] = qps
        logs["remaining"] = remaining
        self.callback_handler.on_log(self.args, self.state, logs=logs)

    def train(self):
        args = self.args
        model = self.model
        dataloader = self.dataloader
        optimizer = self.optimizer
        scheduler = self.scheduler
        scaler = self.scaler
        cleanup_before_training()
        timers = get_timers()
        state = self.state
        num_skips = 0
        accu_loss = torch.zeros(1, dtype=torch.float32, device="cuda")
        accu_grad_norm = torch.zeros(1, dtype=torch.float32, device="cuda")

        self.callback_handler.on_train_begin(args, state)
        timers("interval-time", log_level=0).start(barrier=True)

        logger.info("Start training.")
        model.train()
        start_epoch = 0 if self.sampler is None else self.sampler.epoch
        for epoch in range(start_epoch, self.args.num_epochs):
            epoch_steps = len(dataloader) // self.args.gradient_accumulation_steps
            train_data_iterator = iter(dataloader)
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            state.epoch = epoch
            for _ in range(epoch_steps):
                self.callback_handler.on_step_begin(args, state)
                loss, grad_norm, suc = self.train_step(
                    model, train_data_iterator, optimizer, scheduler, scaler
                )
                accu_loss.add_(loss)
                accu_grad_norm.add_(grad_norm)
                state.global_step += 1
                num_skips += 1 - int(suc)
                self.callback_handler.on_step_end(args, state)
                if state.global_step % args.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    grad_scale = scaler.get_scale()
                    self.log(accu_loss, accu_grad_norm, lr, grad_scale, num_skips)
                    accu_loss.zero_()
                    accu_grad_norm.zero_()
                if state.global_step % args.save_interval == 0:
                    self.save_checkpoint()
                    self.callback_handler.on_save(args, state)
                if state.global_step >= self.max_steps:
                    break
            if state.global_step >= self.max_steps:
                break

        if state.global_step % args.save_interval != 0:
            self.save_checkpoint()
        self.callback_handler.on_train_end(args, state)

    def save_path(self):
        if self.args.remote_save is not None:
            to_remote = True
            uri = self.args.remote_save
        else:
            to_remote = False
            uri = self.args.save
        return uri, to_remote

    def resume_path(self):
        if self.args.remote_load is not None:
            from_remote = True
            path = self.args.remote_load
        elif self.args.load is not None:
            from_remote = False
            path = self.args.load
        else:
            path, from_remote = None, False

        if path is None:
            uri = None
        elif is_mos_uri(path):
            uri = path if is_leaf_uri(path) else get_latest_uri(path)
        else:
            uri = path if is_leaf_ckpt_dir(path) else find_latest_ckpt(path)

        return uri, from_remote

    def save_checkpoint(self):
        pass

    def resume_checkpoint(self):
        pass

    def reset_data_state(self):
        gbs = self.args.global_train_batch_size
        if self.sampler is not None:
            cur_epoch = self.state.global_step // self.steps_per_epoch
            latest_epoch_steps = self.state.global_step % self.steps_per_epoch
            consumed_samples = gbs * latest_epoch_steps
            self.sampler.set_epoch(cur_epoch)
            self.sampler.reset_data_ptr(consumed_samples)
        else:
            consumed_samples = gbs * self.state.global_step
            self.dataset.reset_data_ptr(consumed_samples)
