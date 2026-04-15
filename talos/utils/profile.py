#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2023/12/04 23:49:28
# Author: Shilei Liu

# Adapted from the following file:
# https://code.alibaba-inc.com/openlm/Megatron-LLaMA/blob/features/liushi/65/megatron/profiler.py
from pathlib import Path

from torch.profiler import ProfilerActivity, profile, schedule

from talos.dist import is_world_first_process
from talos.utils import get_logger


__all__ = ["WrappedProfile"]

logger = get_logger()


class WrappedProfile(profile):
    def __init__(
        self,
        output_path: str,
        enable: bool,
        profile_step_start: int = 80,
        with_stack: bool = False,
    ):
        self.output_path = output_path
        self.enable = enable and is_world_first_process()
        self.is_done = False
        if self.enable:
            handler = self._get_trace_handler()
            super().__init__(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    skip_first=profile_step_start, wait=1, warmup=1, active=2, repeat=1
                ),
                with_stack=with_stack,
                with_modules=True,
                on_trace_ready=handler,
            )
            logger.warning("Tracing...")

    def _get_trace_handler(self):
        def _handler(prof: profile):
            if is_world_first_process():
                output_path = Path(self.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(self.output_path)
                logger.warning(f"Export profile trace to {self.output_path}")
            self.is_done = True

        return _handler

    def __enter__(self):
        if self.enable:
            super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            super().__exit__(exc_type, exc_val, exc_tb)

    def step(self):
        if self.enable and not self.is_done:
            super().step()
