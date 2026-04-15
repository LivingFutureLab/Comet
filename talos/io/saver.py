#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/08/24 11:15:36
# Author: Shilei Liu
import json
import os
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist

from talos.args import CkptArguments
from talos.dist import (
    get_local_rank,
    is_local_first_process,
    is_world_first_process,
)
from talos.io.cloud import push
from talos.io.filesystem import get_fs_and_access_path
from talos.io.mos import register_ckpt
from talos.io.paths import (
    TRAIN_STATE_NAME,
    get_rng_state_path,
)
from talos.utils import TrainState, get_logger


logger = get_logger()


__all__ = [
    "get_rng_state",
    "torch_save",
    "CkptSaver",
]


def get_rng_state():
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        if get_local_rank() == -1:
            rng_state["cuda"] = torch.cuda.random.get_rng_state_all()
        else:
            rng_state["cuda"] = torch.cuda.random.get_rng_state()
    return rng_state


def torch_save(obj: object, path: str):
    fs, path = get_fs_and_access_path(path)
    with fs.open(path, "wb") as f:
        torch.save(obj, f)


class CkptSaver:
    def __init__(self, args: CkptArguments) -> None:
        self.args = args

    def _save_rng_state(self, dir: str):
        rng_states = get_rng_state()
        path = get_rng_state_path(dir)
        torch_save(rng_states, path)

    def _save_train_state(self, dir: str, state: TrainState):
        if is_world_first_process():
            path = os.path.join(dir, TRAIN_STATE_NAME)
            with open(path, "w") as f:
                f.write(json.dumps(state.to_dict(), ensure_ascii=True))

    @staticmethod
    def _walk(dir: str):
        filenames = []
        dir = dir.rstrip("/")
        prefix_length = len(dir)
        for root, _, files in os.walk(dir):
            for file in files:
                p = os.path.join(root, file)[prefix_length + 1 :]
                filenames.append(p)
        return filenames

    def _copy_buffer_to_dst(self, src: str, dst: str, ckpt_id: str):
        if is_local_first_process():
            filenames = self._walk(src)
            num_files = len(filenames)
            if num_files >= 64:
                filenames = filenames[:64] + ["..."]
            logger.info(f"Pushing following {num_files} files to cloud: {filenames}")
            result = push(src, ckpt_id, remote_uri=dst, register_ckpt=False)
            logger.info("Pushing done.")
        else:
            result = None
        dist.barrier()
        if is_world_first_process() and result is not None:
            register_ckpt(result)

    @staticmethod
    def _make_dirs(dir: str):
        os.makedirs(dir, exist_ok=True)
        os.makedirs(os.path.join(dir, "rng_states"), exist_ok=True)

    def _pre_make_dirs(self, dir: str, ckpt_id: str, to_remote: bool):
        if to_remote:
            buf = "/tmp/fast_save_buf"
            if is_local_first_process():
                if os.path.exists(buf):
                    shutil.rmtree(buf)
                self._make_dirs(buf)
            dist.barrier()
        else:
            buf = os.path.join(dir, ckpt_id)
            if buf.startswith("/data/oss_bucket"):
                # In the mounted OSS file system, creating folders
                # is not process-safe.
                if is_world_first_process():
                    self._make_dirs(buf)
                dist.barrier()
            else:
                self._make_dirs(buf)
        return buf
