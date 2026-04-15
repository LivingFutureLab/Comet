#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/10/04 14:38:57
# Author: Shilei Liu
from typing import Optional

import torch
import torch.distributed as dist


__all__ = ["async_send_with_grad", "async_recv_with_grad"]


class _AsyncP2PSend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dst: int, group: dist.ProcessGroup):
        ctx.dst = dst
        ctx.group = group
        if group is None:
            kwargs = {"dst": dst}
        else:
            kwargs = {"group_dst": dst, "group": group}
        ctx.handle = dist.isend(tensor, **kwargs)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ctx.handle.wait()
        if ctx.group is None:
            kwargs = {"src": ctx.dst}
        else:
            kwargs = {"group_src": ctx.dst, "group": ctx.group}
        grad_from_dst = torch.empty_like(grad_output)
        dist.recv(grad_from_dst, **kwargs)
        grad = grad_output + grad_from_dst
        return grad, None, None


class _AsyncP2PRecv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, src: int, group: dist.ProcessGroup):
        ctx.src = src
        ctx.group = group
        if group is None:
            kwargs = {"src": src}
        else:
            kwargs = {"group_src": src, "group": group}
        dist.recv(tensor, **kwargs)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Sends the gradient back to the previous rank.
        Ensures the gradient tensor is contiguous before sending.
        """
        grad_output = grad_output.contiguous()
        if ctx.group is None:
            kwargs = {"dst": ctx.src}
        else:
            kwargs = {"group_dst": ctx.src, "group": ctx.group}
        dist.isend(grad_output, **kwargs)
        return None, None, None


def async_send_with_grad(
    tensor: torch.Tensor, dst: int, group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """
    Asynchronously sends a tensor to a destination rank with autograd support.

    This function is a differentiable wrapper around `torch.distributed.isend`.

    - In the forward pass, it performs a non-blocking send and immediately
      returns the original tensor, acting as an identity function in the
      local computation graph.
    - In the backward pass, it first waits for the forward send to complete,
      then performs a blocking receive to get the gradient from the destination
      rank, and finally adds this remote gradient to the local gradient.

    Args:
        tensor (torch.Tensor): The tensor to send.
        dst (int): The destination rank ID (global rank or rank on group).
        group (Optional[dist.ProcessGroup]): The process group to work on. If `None`,
            the default global process group is used.

    Returns:
        torch.Tensor: The input tensor itself, allowing for chaining operations or
            continued use in the local computation graph.
    """
    return _AsyncP2PSend.apply(tensor, dst, group)


def async_recv_with_grad(
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    src: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Receives a tensor from a source rank with autograd support.

    This function is a differentiable wrapper around `torch.distributed.recv`.

    - In the forward pass, it creates a placeholder tensor, then performs a
      blocking receive to populate it with data from the source rank.
    - In the backward pass, it performs a non-blocking send to send the
      gradient of the received tensor back to the source rank.

    Args:
        shape (Tuple[int, ...]): The shape of the tensor to receive.
        dtype (torch.dtype): The data type of the tensor to receive.
        device (torch.device): The device where the tensor should be created.
        src (int): The source rank ID (global rank or rank on group).
        group (Optional[dist.ProcessGroup]): The process group to work on. If `None`,
            the default global process group is used.

    Returns:
        torch.Tensor: The received tensor, which is connected to the current
            computation graph.
    """
    # A placeholder is created with `requires_grad=True`. This is crucial because
    # it ensures the output tensor is part of the autograd graph from the start,
    # allowing it to capture gradients and trigger the backward pass correctly.
    placeholder = torch.empty(shape, dtype=dtype, device=device, requires_grad=True)
    return _AsyncP2PRecv.apply(placeholder, src, group)
