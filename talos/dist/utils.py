#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/09 22:41:37
# Author: Shilei Liu
import os
from typing import Optional, List

import torch.distributed as dist
from torch.distributed import ProcessGroup

__all__ = [
    "get_local_rank",
    "get_global_rank",
    "get_world_size",
    "is_local_first_process",
    "is_world_first_process",
    "get_world_size_per_node",
    "get_global_ranks_on_local_node",
    "get_master_addr",
    "get_master_port",
    "get_worker_id",
]


def get_local_rank() -> int:
    """Gets the local rank of the current process.

    Reads from the 'LOCAL_RANK' environment variable. Defaults to -1 if not set.

    Returns:
        int: The local rank.
    """
    return int(os.getenv("LOCAL_RANK", "-1"))


def get_global_rank() -> int:
    """Gets the global rank of the current process.

    Reads from the 'RANK' environment variable. Defaults to 0 if not set.

    Returns:
        int: The global rank.
    """
    return int(os.getenv("RANK", "0"))


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Gets the total number of processes in the distributed group.

    If a process group is provided, it returns the size of that group.
    Otherwise, it reads from the 'WORLD_SIZE' environment variable.
    Defaults to 1 if not set.

    Args:
        group (Optional[ProcessGroup]): The process group. Defaults to None.

    Returns:
        int: The world size.
    """
    if group is not None:
        return dist.get_world_size(group)
    return int(os.getenv("WORLD_SIZE", "1"))


def is_local_first_process() -> bool:
    """Checks if the current process is the main process on the local node (local_rank == 0).

    Returns True for non-distributed scenarios (local_rank == -1).

    Returns:
        bool: True if it's the main local process.
    """
    
    if (local_rank := int(os.getenv("LOCAL_PROCESS_RANK", "-1"))) != -1:
        return local_rank == 0
    else:
        return get_local_rank() in (-1, 0)


def is_world_first_process() -> bool:
    """Checks if the current process is the main process across all nodes (global_rank == 0).

    Returns True for non-distributed scenarios.

    Returns:
        bool: True if it's the main global process.
    """
    return get_local_rank() == -1 or get_global_rank() in (-1, 0)


def get_world_size_per_node() -> int:
    """Determines the number of processes/devices per node.

    It tries to infer this value from the following environment variables in order:
    1. `WORLD_SIZE` and `NODES`: `WORLD_SIZE / NODES`
    2. `NPROC_PER_NODE`
    3. `NVIDIA_VISIBLE_DEVICES` (by counting the devices)

    Raises:
        RuntimeError: If the number of devices per node cannot be determined.

    Returns:
        int: The number of devices per node.
    """
    # 1. From world size and node count
    world_size = get_world_size()
    nnodes = int(os.getenv("NODES", "-1"))
    if nnodes != -1 and world_size > 0:
        if world_size % nnodes != 0:
            raise RuntimeError(
                f"Cannot evenly divide WORLD_SIZE({world_size}) by NODES({nnodes})."
            )
        return world_size // nnodes

    # 2. From nproc_per_node
    device_per_node = int(os.getenv("NPROC_PER_NODE", "-1"))
    if device_per_node != -1:
        return device_per_node

    # 3. From visible devices
    visible_devices = os.getenv("NVIDIA_VISIBLE_DEVICES")
    if visible_devices is not None:
        return len(visible_devices.split(","))

    if not dist.is_initialized():
        return 1

    raise RuntimeError(
        "Could not determine the number of devices per node. "
        "Please set NPROC_PER_NODE or ensure NVIDIA_VISIBLE_DEVICES is set."
    )


def get_global_ranks_on_local_node() -> List[int]:
    """Gets a list of global ranks for all processes running on the same node as the current process.

    Returns:
        List[int]: A list of global ranks on the local node.
    """
    ranks_per_node = get_world_size_per_node()
    rank = get_global_rank()
    node_rank = rank // ranks_per_node
    start = node_rank * ranks_per_node
    end = (node_rank + 1) * ranks_per_node
    ranks = list(range(start, end))
    return ranks


def get_master_addr() -> str:
    """Gets the address of the master node.

    Reads from 'MASTER_ADDR' environment variable. Defaults to '127.0.0.1'.

    Returns:
        str: The master address.
    """
    return os.getenv("MASTER_ADDR", "127.0.0.1")


def get_master_port() -> str:
    """Gets the port of the master node.

    Reads from 'MASTER_PORT' environment variable. Defaults to '6379'.

    Returns:
        str: The master port.
    """
    return os.getenv("MASTER_PORT", "6379")


def get_worker_id() -> str:
    """Gets a unique identifier for the current worker process.

    Tries to read from 'WORKER_ID'. If not available, it constructs an ID
    using the master address and global rank (e.g., '127.0.0.1:0').

    Returns:
        str: A unique ID for the worker.
    """
    return os.getenv("WORKER_ID", f"{get_master_addr()}:{get_global_rank()}")
