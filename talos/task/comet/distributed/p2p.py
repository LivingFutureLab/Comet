#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/28 00:55:17
# Author: Shilei Liu
from typing import Optional

import torch
import torch.distributed as dist


__all__ = ["send_with_grad", "recv_with_grad"]


class _P2PSend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dst: int, group: dist.ProcessGroup):
        ctx.dst = dst
        ctx.group = group
        if group is None:
            kwargs = {"dst": dst}
        else:
            kwargs = {"group_dst": dst, "group": group}
        dist.send(tensor, **kwargs)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.group is None:
            kwargs = {"src": ctx.dst}
        else:
            kwargs = {"group_src": ctx.dst, "group": ctx.group}
        grad_from_dst = torch.empty_like(grad_output)
        dist.recv(grad_from_dst, **kwargs)
        grad = grad_output + grad_from_dst
        return grad, None, None


class _P2PRecv(torch.autograd.Function):
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
        dist.send(grad_output, **kwargs)
        return None, None, None


def send_with_grad(
    tensor: torch.Tensor, dst: int, group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """
    Sends a tensor to a destination rank and enables gradient backpropagation.

    This function is a key component for pipeline parallelism. It sends a tensor
    to the next stage in the pipeline while creating a node in the autograd graph.
    During the backward pass, this node will perform two actions:
    1. Receive the gradient from the destination rank (`dst`).
    2. Add this received gradient to the gradient flowing from the local graph.

    This ensures that the gradient from the downstream pipeline stages is correctly
    propagated back to the upstream stages.

    Args:
        tensor (torch.Tensor): The tensor to send.
        dst (int): The destination rank.
        group (Optional[dist.ProcessGroup]): The process group to work on. If None,
            the default process group will be used.

    Returns:
        torch.Tensor: The same input tensor, now with a `_P2PSend` backward hook attached.
                      This returned tensor **must** be used in subsequent computations
                      to connect the autograd graph correctly.
    """
    return _P2PSend.apply(tensor, dst, group)


def recv_with_grad(
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    src: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Receives a tensor from a source rank and enables gradient backpropagation.

    This function complements `send_with_grad`. It creates a placeholder tensor
    and receives data into it from the source rank. It also creates a node in
    the autograd graph. During the backward pass, this node will send the
    gradient of the received tensor back to the source rank (`src`).

    This allows the gradient to flow from a downstream pipeline stage back to
    an upstream stage.

    Args:
        shape (torch.Size): The shape of the tensor to be received.
        dtype (torch.dtype): The data type of the tensor.
        device (torch.device): The device where the tensor should be created.
        src (int): The source rank from which to receive the tensor.
        group (Optional[dist.ProcessGroup]): The process group to work on. If None,
            the default process group will be used.

    Returns:
        torch.Tensor: The received tensor, with a `_P2PRecv` backward hook attached.
    """
    # A placeholder is created with `requires_grad=True` to ensure it's part of
    # the autograd graph from the beginning.
    placeholder = torch.empty(shape, dtype=dtype, device=device, requires_grad=True)
    return _P2PRecv.apply(placeholder, src, group)
