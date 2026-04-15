#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/28 00:45:31
# Author: Shilei Liu
import torch
import torch.distributed as dist

from talos.task.comet.distributed.parallel_state import ParallelState
from talos.task.comet.utils import CometBatch


__all__ = ["batch_dispatch"]


def batch_dispatch(state: ParallelState, batch: CometBatch) -> CometBatch:
    """
    Dispatches a batch across sequence parallel ranks using the "Pad-and-Scatter"
    strategy for optimal performance.

    This function is a cornerstone of sequence parallelism. It takes a full batch
    on rank 0, splits it along the sequence dimension, and distributes the
    resulting sub-batches to the corresponding ranks in the parallel group.

    - For large core tensors (e.g., `input_ids`), it uses a highly efficient
      "Pad-and-Scatter" method.
    - For small, complex metadata, it uses `scatter_object_list` for simplicity.

    Args:
        state (ParallelState): The distributed state for the sequence parallel group.
        batch (CometBatch): The collated batch, which must be present on rank 0 of the
                      group and can be `None` on other ranks.

    Returns:
        CometBatch: A sub-batch containing only the data assigned to the current rank.
    """
    # Stage 1: Broadcast Metadata
    metadata = {}
    full_chunk_sizes = None
    if state.is_first_rank:
        if batch is None or "chunk_sizes" not in batch:
            raise ValueError("Batch with 'chunk_sizes' must be provided on rank 0.")

        num_chunks = len(batch["chunk_sizes"])
        if num_chunks % state.world_size != 0:
            raise ValueError(
                f"Number of chunks ({num_chunks}) must be divisible by "
                f"sequence parallel world size ({state.world_size})."
            )

        num_local_chunks = num_chunks // state.world_size
        metadata["num_local_chunks"] = num_local_chunks
        metadata["batch_size"] = batch["input_ids"].shape[0]
        metadata["dtype_map"] = {
            k: batch[k].dtype
            for k in ["input_ids", "labels", "attention_mask", "position_ids"]
        }

        full_chunk_sizes = batch["chunk_sizes"]

        local_seq_lens = [
            sum(full_chunk_sizes[i * num_local_chunks : (i + 1) * num_local_chunks])
            for i in range(state.world_size)
        ]
        metadata["local_seq_lens"] = local_seq_lens
        metadata["max_local_seq_len"] = max(local_seq_lens)

    metadata_list = [metadata if state.is_first_rank else None]
    dist.broadcast_object_list(metadata_list, group_src=0, group=state.group)

    if not state.is_first_rank:
        metadata = metadata_list[0]

    device = torch.device("cuda")
    batch_size = metadata["batch_size"]
    dtype_map = metadata["dtype_map"]
    num_local_chunks = metadata["num_local_chunks"]
    max_local_seq_len = metadata["max_local_seq_len"]
    local_seq_lens = metadata["local_seq_lens"]

    local_batch = {}

    # Stage 2
    object_keys = [
        "chunk_sizes",
        "global_mem_inds",
        "global_beacon_inds",
        "temp_mem_inds",
        "temp_beacon_inds",
        "temp_mem_select_inds",
    ]

    for key in object_keys:
        scatter_list = None
        if state.is_first_rank:
            original_list = batch[key]
            if isinstance(original_list[0], torch.Tensor):
                original_list = [t.to(device) for t in original_list]
            scatter_list = [
                original_list[i * num_local_chunks : (i + 1) * num_local_chunks]
                for i in range(state.world_size)
            ]

        output_list = [None]
        dist.scatter_object_list(
            output_list, scatter_list, group_src=0, group=state.group
        )

        output = output_list[0]
        if isinstance(output, torch.Tensor):
            output = output.to(device)
        elif isinstance(output, list) and isinstance(output[0], torch.Tensor):
            output = [t.to(device) for t in output]
        local_batch[key] = output

    # Stage 3: Scatter Core Tensors via "Pad-and-Scatter"
    tensor_keys = ["input_ids", "labels", "position_ids", "attention_mask"]

    local_true_seq_len = local_seq_lens[state.rank]

    for key in tensor_keys:
        scatter_list = None
        if state.is_first_rank:
            full_tensor = batch[key].to(device=device)
            scatter_list = []

            chunk_start_indices = [0] * (len(full_chunk_sizes) + 1)
            for i in range(len(full_chunk_sizes)):
                chunk_start_indices[i + 1] = (
                    chunk_start_indices[i] + full_chunk_sizes[i]
                )

            for i in range(state.world_size):
                # 1. Slice the full tensor to get the data for rank `i`.
                start_chunk_idx = i * num_local_chunks
                end_chunk_idx = (i + 1) * num_local_chunks
                start_pos = chunk_start_indices[start_chunk_idx]
                end_pos = chunk_start_indices[end_chunk_idx]
                tensor_slice = full_tensor[:, start_pos:end_pos]

                # 2. Pad the slice to the maximum local sequence length.
                diff = max_local_seq_len - tensor_slice.shape[1]
                if diff > 0:
                    padding = tensor_slice.new_zeros(batch_size, diff)
                    padded_slice = torch.cat([tensor_slice, padding], dim=1)
                else:
                    padded_slice = tensor_slice

                scatter_list.append(padded_slice.contiguous())

        # All ranks prepare a receive buffer of the same maximum size.
        recv_tensor = torch.empty(
            (batch_size, max_local_seq_len),
            dtype=dtype_map[key],
            device=device,
        )

        # 3. Perform the highly optimized scatter operation.
        dist.scatter(recv_tensor, scatter_list, group_src=0, group=state.group)

        # 4. Unpad the received tensor to its true local length.
        if max_local_seq_len > local_true_seq_len:
            local_batch[key] = recv_tensor[:, :local_true_seq_len].contiguous()
        else:
            local_batch[key] = recv_tensor

    return local_batch
