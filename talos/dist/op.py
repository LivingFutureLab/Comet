#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/11/23 17:53:00
# Author: Shilei Liu
from abc import ABC, abstractmethod
from typing import Optional, Union
import torch
import torch.distributed as dist


__all__ = [
    "TensorSplitter",
    "RingSplitter",
    "ContextParallelismBatchScatter",
    "broadcast_string",
]


class TensorSplitter(ABC):
    """
    Abstract base class defining the interface for splitting a single tensor
    into a list of chunks.
    """

    def __init__(self, world_size: int, dim: int):
        self.world_size = world_size
        self.dim = dim

    @abstractmethod
    def split(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        The specific splitting logic, which must be implemented by subclasses.

        Args:
            tensor (torch.Tensor): The input tensor to be split.

        Returns:
            list[torch.Tensor]: A list of tensor chunks, where the i-th element
                                is intended for rank i.
        """
        pass


class RingSplitter(TensorSplitter):
    """Implements the 'ring' splitting strategy."""

    def split(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        Splits the tensor into 2 * world_size chunks along the specified dimension.
        The first half is kept in order, while the second half is reversed and
        then concatenated.
        """
        dim = self.dim
        if self.dim < 0:
            dim = tensor.dim() + dim

        dim_size = tensor.shape[dim]
        num_chunks = 2 * self.world_size
        if dim_size % num_chunks != 0:
            raise ValueError(
                f"Dimension {dim} with size {dim_size} cannot be evenly "
                f"split into {num_chunks} chunks."
            )
        chunk_size = dim_size // num_chunks

        original_shape = list(tensor.shape)
        new_shape = (
            original_shape[:dim]
            + [2, self.world_size, chunk_size]
            + original_shape[dim + 1 :]
        )
        viewed_tensor = tensor.view(new_shape)

        # Move the '2' (front/back) and 'world_size' dimensions to the front for easier manipulation.
        permuted_tensor = torch.movedim(viewed_tensor, [dim, dim + 1], [0, 1])

        front_half = permuted_tensor[0]
        back_half = permuted_tensor[1]

        # Reverse the back half along the world_size dimension (now at dim=0).
        back_half_reversed = torch.flip(back_half, dims=[0])

        # Concatenate along the last dimension, which corresponds to the chunk_size
        # of the original split dimension.
        output_tensor = torch.cat([front_half, back_half_reversed], dim=-1)

        # Unbind along dim=0 to split the [world_size, ...] tensor into a list
        # of world_size tensors.
        return list(output_tensor.unbind(0))


class ContextParallelismBatchScatter:
    """
    Handles the splitting and scattering of a batch of tensors (a dictionary)
    from a source rank (rank 0) to all ranks in a process group.
    """

    def __init__(self, group: dist.ProcessGroup, device: torch.device, dim: int = 1):
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.src_rank = 0

        # Instantiate the desired splitting strategy.
        self.splitter = RingSplitter(self.world_size, dim)

    def __call__(
        self, batch: Optional[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """
        Executes the splitting and scattering operation.

        Args:
            batch (Optional[dict[str, torch.Tensor]]): A dictionary of tensors on the
                source rank. Should be None on non-source ranks.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the local chunk of each
                tensor for the current rank.
        """
        scatter_lists: dict[str, list[torch.Tensor]] = {}
        is_src_rank = batch is not None

        if is_src_rank:
            metadatas = []
            for key, tensor in batch.items():
                scatter_list = self.splitter.split(tensor.to(self.device))
                scatter_lists[key] = scatter_list

                # The shape and dtype of the chunk for rank 0 is representative for all ranks.
                local_chunk = scatter_list[0]

                metadata = {
                    "key": key,
                    "dtype": local_chunk.dtype,
                    "local_shape": local_chunk.shape,
                }
                metadatas.append(metadata)
        else:
            metadatas = None

        broadcast_obj_list = [metadatas]
        dist.broadcast_object_list(
            broadcast_obj_list, group_src=self.src_rank, group=self.group
        )
        metadatas = broadcast_obj_list[0]

        local_batch = {}
        for metadata in metadatas:
            key = metadata["key"]

            local_tensor = torch.empty(
                metadata["local_shape"], dtype=metadata["dtype"], device=self.device
            )

            scatter_list = scatter_lists.get(key) if is_src_rank else None
            dist.scatter(
                tensor=local_tensor,
                scatter_list=scatter_list,
                group_src=self.src_rank,
                group=self.group,
            )
            local_batch[key] = local_tensor

        return local_batch


def broadcast_string(
    s: str,
    src: int,
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[Union[torch.device, str, int]] = None,
) -> str:
    """Broadcasts a string from a source rank to all other ranks in a process group.

    Since `torch.distributed.broadcast` only operates on tensors, this function
    implements a two-step process to broadcast a string of variable length:
    1. The source rank encodes the string into a uint8 tensor, and its length
       is broadcast to all other ranks.
    2. Other ranks create an empty tensor of the received length to serve as a
       buffer.
    3. The actual tensor containing the encoded string is then broadcast.
    4. All ranks decode the tensor back into a string.

    Args:
        s (str): The string to be broadcast. This value is only used on the `src`
            rank; its value on other ranks is ignored.
        src (int): The rank of the process that is sending the string.
        group (Optional[dist.ProcessGroup]): The process group to work on. If
            `None`, the default process group is used.
        device (Optional[Union[torch.device, str, int]]): The device to use for
            the intermediate tensors used in broadcasting. If `None`, it defaults
            to 'cuda' if available, otherwise 'cpu'.

    Returns:
        str: The string received from the `src` rank. On the `src` rank, this
             is the same as the input string `s`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = dist.get_rank(group)

    # Step 1: Broadcast the length of the string
    if rank == src:
        encoded = s.encode("utf-8")
        encoded_tensor = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
        len_tensor = torch.tensor(
            [len(encoded_tensor)], dtype=torch.long, device=device
        )
    else:
        # Other ranks create a placeholder tensor to receive the length
        len_tensor = torch.zeros(1, dtype=torch.long, device=device)

    dist.broadcast(len_tensor, src=src, group=group)

    str_len = len_tensor.item()

    # Step 2: Broadcast the string content
    if rank != src:
        # Other ranks create a buffer of the correct size
        encoded_tensor = torch.empty(str_len, dtype=torch.uint8, device=device)

    dist.broadcast(encoded_tensor, src=src, group=group)

    # Decode the tensor back to a string
    received_bytes = bytes(encoded_tensor.cpu().tolist())
    received_string = received_bytes.decode("utf-8")

    return received_string
