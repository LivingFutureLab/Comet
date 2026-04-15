#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/11 00:31:01
# Author: Shilei Liu
import datetime
from typing import Any, List

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from talos.args import TrainingArguments
from talos.dist import is_local_first_process, is_world_first_process
from talos.utils import (
    TrackerPersistentState,
    TrainState,
    WrappedProfile,
    format_time,
    get_logger,
    get_timers,
)


logger = get_logger()

__all__ = [
    "TrainerCallback",
    "CallbackHandler",
    "TrackerCallback",
    "LogCallback",
    "ProfileCallback",
    "default_callbacks",
]


class TrainerCallback:
    def on_init_end(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass

    def on_log(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass

    def on_save(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass

    def on_substep_begin(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainState, **kwargs):
        pass


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(
        self,
        callbacks: List[TrainerCallback],
        model: Module,
        tokenizer: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: ShardedGradScaler,
    ):
        self.callbacks = callbacks
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

    def callback_list(self):
        return [cb.__class__.__name__ for cb in self.callbacks]

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainState,
    ):
        self.call_event("on_init_end", args, state)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainState,
    ):
        self.call_event("on_train_begin", args, state)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainState,
    ):
        self.call_event("on_train_end", args, state)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainState,
        logs: dict[str, Any],
    ):
        self.call_event("on_log", args, state, logs=logs)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainState,
    ):
        self.call_event("on_save", args, state)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainState,
    ):
        self.call_event("on_step_begin", args, state)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainState,
    ):
        self.call_event("on_step_end", args, state)

    def on_substep_begin(
        self,
        args: TrainingArguments,
        state: TrainState,
    ):
        self.call_event("on_substep_begin", args, state)

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainState,
    ):
        self.call_event("on_substep_end", args, state)

    def call_event(
        self, event: str, args: TrainingArguments, state: TrainState, **kwargs
    ):
        for callback in self.callbacks:
            getattr(callback, event)(
                args,
                state,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                **kwargs,
            )


class TrackerCallback(TrainerCallback):
    def __init__(self, args: TrainingArguments, prefix: str = "train"):
        if args.tracker_impl != "none" and is_world_first_process():
            self.tracker = TrackerPersistentState.build(
                args.tracker_impl,
                args.tracker_proj,
                args.tracker_runid,
                args.tracker_log_dir,
                args.to_dict(),
            )
        else:
            self.tracker = None
        self.prefix = prefix

    def on_log(
        self, args: TrainingArguments, state: TrainState, logs: dict[str, Any], **kwargs
    ):
        if self.tracker is None:
            return
        metrics = {
            f"{self.prefix}/{k}": v
            for k, v in logs.items()
            if isinstance(v, (int, float))
        }
        self.tracker.log(metrics, step=state.global_step)

    def on_train_end(self, args: TrainingArguments, state: TrainState, **kwargs):
        if self.tracker is not None:
            self.tracker.close()


class LogCallback(TrainerCallback):
    def timers_log(self, log_interval: int):
        names_to_log = [
            "batch-generator",
            "forward-backward",
            "forward-compute",
            "backward-compute",
            "optimizer",
        ]
        timers = get_timers()
        timers.log(names_to_log, normalizer=log_interval)

    def default_logs_to_str(
        self,
        args: TrainingArguments,
        state: TrainState,
        logs: dict[str, Any],
    ):
        default_keys = [
            "loss",
            "lr",
            "grad_scale",
            "grad_norm",
            "num_skips",
            "consumed_samples",
            "qps",
            "remaining",
        ]
        global_step = state.global_step
        epoch = state.epoch
        loss = logs["loss"]
        lr = logs["lr"]
        grad_scale = logs["grad_scale"]
        grad_norm = logs["grad_norm"]
        num_skips = logs["num_skips"]
        consumed_samples = logs["consumed_samples"]
        qps = logs["qps"]
        remaining = format_time(logs["remaining"])

        msg = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] "
        msg += f"step: {global_step} / {state.max_steps}, "
        if args.num_epochs > 1:
            msg += f"epoch: {epoch}, "
        msg += f"loss: {loss:.6f}, lr: {lr:6f}, "
        if args.dtype == "fp16":
            msg += f"grad_scale: {grad_scale}, "
        msg += f"grad_norm: {grad_norm:.2f}, "
        msg += f"num_skips: {num_skips}, "
        for k, v in logs.items():
            if k in default_keys:
                continue
            if isinstance(v, float):
                v = f"{v:.3f}"
            msg += f"{k}: {v}, "
        msg += f"consumed_samples: {consumed_samples}, "
        msg += f"qps: {qps:.2f}, "
        msg += f"remaining: {remaining}"
        return msg

    def on_log(
        self, args: TrainingArguments, state: TrainState, logs: dict[str, Any], **kwargs
    ):
        msg = self.default_logs_to_str(args, state, logs)

        if is_local_first_process():
            print(msg)

        self.timers_log(args.log_interval)
        state.log_history.clear()


class ProfileCallback(TrainerCallback):
    def __init__(
        self,
        trace_path: str,
        profile_step_start: int = 80,
        with_stack: bool = False,
    ):
        self.trace_path = trace_path
        self.profile_step_start = profile_step_start
        self.with_stack = with_stack
        self.profiler = None

    def on_train_begin(self, args, state, **kwargs):
        self.profiler = WrappedProfile(
            output_path=self.trace_path,
            enable=True,
            profile_step_start=self.profile_step_start,
            with_stack=self.with_stack,
        )
        if self.profiler.enable:
            self.profiler.__enter__()

    def on_step_end(self, args, state, **kwargs):
        self.profiler.step()

    def on_train_end(self, args, state, **kwargs):
        self.profiler.__exit__(None, None, None)


def default_callbacks(args: TrainingArguments):
    callbacks = [
        LogCallback(),
        TrackerCallback(args),
    ]
    if args.do_profile:
        callbacks.append(ProfileCallback(args.trace_path))
    return callbacks
