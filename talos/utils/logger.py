#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/09 22:50:53
# Author: Shilei Liu
import logging
from typing import Optional

from talos.dist import is_local_first_process


__all__ = ["LoggerPersistentState", "get_logger"]


class LoggerPersistentState:
    logger: Optional[logging.Logger] = None

    @classmethod
    def init(cls):
        logger = logging.getLogger("TaoScale")

        fmt = "%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] %(message)s"
        datefmt = "%m/%d/%Y %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        level = logging.INFO if is_local_first_process() else logging.WARNING
        logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        LoggerPersistentState.set(logger)

    @classmethod
    def set(cls, logger):
        cls.logger = logger

    @classmethod
    def get(cls):
        return cls.logger

    def __new__(cls):
        raise TypeError("Namespace class, non-instantiable")


LoggerPersistentState.init()


def get_logger():
    return LoggerPersistentState.get()
