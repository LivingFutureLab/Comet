#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/16 16:06:12
# Author: Shilei Liu
from typing import Any

import torch

from talos.io.hf_loader import HFCkptLoader
from talos.io.hf_saver import HFCkptSaver
from talos.train.trainer import Trainer
from talos.utils import get_logger


__all__ = ["HFTrainer", "HFCustomLogsTrainer"]


logger = get_logger()


class HFTrainer(Trainer):
    def resume_checkpoint(self):
        load_dir, from_remote = self.resume_path()
        if load_dir is None:
            logger.info("Can not find checkpoint, skip resume.")
            return
        logger.info(f"Resume training from {load_dir}")
        loader = HFCkptLoader(self.args)
        loader.load(
            load_dir,
            self.state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            from_remote,
        )
        self.reset_data_state()

    def save_checkpoint(self):
        step = self.state.global_step
        save_dir, to_remote = self.save_path()
        ckpt_id = f"checkpoint-{step}"
        saver = HFCkptSaver(self.args)
        saver.save(
            save_dir,
            ckpt_id,
            self.state,
            self.tokenizer,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            to_remote,
        )


class HFCustomLogsTrainer(HFTrainer):
    def log(self, accu_loss, accu_grad_norm, lr, grad_scale, num_skips, logs=None):
        state = self.state
        interval = self.args.log_interval

        hist = state.log_history[0]

        logs = {}
        for k, v in hist.items():
            v = self.all_reduce_avg(v).item() / interval
            logs[k] = v

        super().log(accu_loss, accu_grad_norm, lr, grad_scale, num_skips, logs)

    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]):
        state = self.state
        outputs = model(**batch)
        loss = outputs.pop("loss")
        metrics = {}
        for k, v in outputs.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=loss.device, dtype=torch.float32)
            # Cast to float32 for finer-grained logging (metrics may be bfloat16).
            v = v.detach().float() / self.args.gradient_accumulation_steps
            metrics[k] = v

        if len(state.log_history) == 0:
            state.log_history.append(metrics)
        else:
            for k, v in metrics.items():
                state.log_history[0][k].add_(v)
        return loss
