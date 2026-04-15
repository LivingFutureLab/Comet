#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/29 14:41:31
# Author: Shilei Liu
import os
import signal
import traceback

import torch.multiprocessing as mp

from talos.init import cleanup, setup
from talos.utils.logger import get_logger
from talos.utils.tools import set_seed


__all__ = ["DistributedTestSignalHandler", "DistributedTestBase"]

logger = get_logger()


class DistributedTestSignalHandler:
    def __init__(self):
        self.original_handlers = {}

    def _handler(self, signum, frame):
        logger.warning(f"PID {os.getpid()} received signal {signum}. Forcing exit.")
        os._exit(1)

    def __enter__(self):
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]
        for sig in signals_to_handle:
            self.original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handler)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        # This part handles NON-SIGNAL exceptions (e.g., AssertionError).
        # In this case, processes are responsive, so a graceful cleanup is correct.
        for sig, handler in self.original_handlers.items():
            signal.signal(sig, handler)
        if exc_type is not None:
            cleanup()


class DistributedTestBase:
    """
    A base class for creating multi-process distributed tests compatible with pytest.

    Inherit from this class and define your test logic in methods. Then, create
    `test_*` methods that call `self.run_test("your_logic_method_name")`.
    """

    world_size: int = 2
    port: int = 12358
    seed: int = 42

    @staticmethod
    def setup_dist(rank: int, world_size: int, port: int, seed: int = 42):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        setup()
        set_seed(seed)

    def _worker_fn(self, rank: int, test_method_name: str, *args):
        """
        The actual worker function that runs in each spawned process.
        It sets up the environment, creates an instance of the test class,
        and runs the specified test method.
        """
        with DistributedTestSignalHandler():
            try:
                self.setup_dist(rank, self.world_size, self.port, self.seed)

                # Each process needs its own instance of the test class.
                # `type(self)` gets the child class (e.g., TestMyComponent).
                instance = type(self)()

                # Use getattr to find the method by name and execute it.
                test_to_run = getattr(instance, test_method_name)

                # Check if the method needs rank and world_size args.
                test_to_run(rank, self.world_size, *args)
            except Exception:
                traceback.print_exc()
                os._exit(1)
            finally:
                cleanup()

    def run_test(self, test_method_name: str, *args):
        """
        The main entry point to launch a distributed test.
        This method is called from within a `test_*` method in the child class.
        """
        # Ensure the method to be tested actually exists on the class.
        if not hasattr(self, test_method_name):
            raise AttributeError(
                f"{type(self).__name__} does not have a method named '{test_method_name}'"
            )

        mp.spawn(
            self._worker_fn,
            args=(test_method_name, *args),
            nprocs=self.world_size,
            join=True,
        )
