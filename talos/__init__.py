#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/09 22:40:44
# Author: Shilei Liu
from typing import TYPE_CHECKING

import talos.data as data
import talos.dist as dist
import talos.init as init
import talos.io as io
import talos.optim as optim
import talos.utils as utils
from talos.utils.lazy_loader import LazyLoader


if TYPE_CHECKING:
    import talos.fsdp as fsdp
    import talos.train as train
else:
    fsdp = LazyLoader("fsdp", globals(), "talos.fsdp")
    train = LazyLoader("train", globals(), "talos.train")


__version__ = "0.0.1+comet.opensource"
