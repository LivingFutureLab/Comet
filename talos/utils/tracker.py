#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/10 10:59:10
# Author: Shilei Liu
import re
from typing import Any, Optional


__all__ = [
    "BaseTracker",
    "MLTracker",
    "TensorBoardTracker",
    "TrackerPersistentState",
    "get_tracker",
]


class BaseTracker:
    def log(self, metrics: dict[str, Any], step: Optional[int] = None):
        pass

    def close(self):
        pass


class MLTracker(BaseTracker):
    def __init__(
        self,
        project: str,
        run_id: str,
        entity: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
    ):
        import ml_tracker

        self.check(project, run_id, entity)
        self.tracker = ml_tracker
        self.tracker.init(project=project, entity=entity, id=run_id, config=args)

    def log(self, metrics: dict[str, Any], step: Optional[int] = None):
        self.tracker.log(metrics, step)

    def close(self):
        self.tracker.finish()

    def check(self, project: str, run_id: str, entity: Optional[str] = None):
        assert project is not None
        assert run_id is not None
        if entity is not None:
            assert re.fullmatch(r"[a-zA-Z0-9_.]+", entity) is not None


class TensorBoardTracker(BaseTracker):
    def __init__(self, log_dir: str, args: Optional[dict[str, Any]] = None):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir, max_queue=1000, flush_secs=10)
        if args is not None:
            config = {
                k: v for k, v in args.items() if isinstance(v, (int, float, str, bool))
            }
            self.writer.add_hparams(hparam_dict=config, metric_dict={})
        self.writer.flush()

    def log(self, metrics: dict[str, Any], step: Optional[int] = None):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step)
        self.writer.flush()

    def close(self):
        self.writer.close()


class TrackerPersistentState:
    tracker: Optional[BaseTracker] = None

    @classmethod
    def build(
        cls,
        impl: str,
        project: Optional[str] = None,
        run_id: Optional[str] = None,
        log_dir: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
    ):
        if impl == "ml_tracker":
            tracker = MLTracker(project, run_id, entity=log_dir, args=args)
        elif impl == "tensorboard":
            assert log_dir is not None
            tracker = TensorBoardTracker(log_dir)
        else:
            raise NotImplementedError(f"No tracker implementation `{impl}`")
        return tracker

    @classmethod
    def init(
        cls,
        impl: str,
        project: Optional[str] = None,
        run_id: Optional[str] = None,
        log_dir: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
    ):
        tracker = TrackerPersistentState.build(impl, project, run_id, log_dir, args)
        TrackerPersistentState.set(tracker)

    @classmethod
    def set(cls, tracker):
        cls.tracker = tracker

    @classmethod
    def get(cls):
        return cls.tracker

    def __new__(cls):
        raise TypeError("Namespace class, non-instantiable")


def get_tracker() -> BaseTracker:
    return TrackerPersistentState.get()
