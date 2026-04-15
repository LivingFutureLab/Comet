#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2026/04/15 11:53:57
# Author: Guangshu
import fused_weight_gradient_mlp_cuda
import torch
import triton
from quack.cross_entropy import cross_entropy_fwd_out


class ChunkedLinearCrossEntropyLossFunction(torch.autograd.Function):
    # NOTE: We put the linear projection in the same autograd Function as the loss computation
    # because we overwrite the logits with their gradients inplace to avoid allocating more
    # memory for the gradients, and so we keep the logits completely contained within this
    # Functionto avoid possible side-effects if they were exposed.

    @staticmethod
    def forward(
        ctx,
        in_feat: torch.Tensor,
        proj_weight: torch.Tensor,
        target: torch.Tensor,
        grad_proj_weight: torch.Tensor,
        n_loop_iters: int,
        ignore_index: int,
        reduction: str,
    ):
        n_tokens = in_feat.shape[0]
        n_classes = proj_weight.shape[0]

        assert in_feat.ndim == 2, in_feat.ndim
        assert proj_weight.ndim == 2, proj_weight.ndim
        assert target.ndim == 1, target.shape
        assert in_feat.shape[0] == target.shape[0], (
            f"Number of tokens in in_feat and targ is not equal: {(in_feat.shape, target.shape) = }"
        )
        assert reduction in ("mean", "sum"), reduction
        assert n_loop_iters > 0, n_loop_iters
        assert n_tokens % n_loop_iters == 0, (n_tokens, n_loop_iters)

        loss = torch.empty(n_tokens, dtype=torch.float32, device=in_feat.device)
        dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else in_feat.dtype
        )

        grad_in_feat = (
            torch.empty_like(in_feat, dtype=dtype) if in_feat.requires_grad else None
        )
        if proj_weight.requires_grad and grad_proj_weight is None:
            grad_proj_weight = torch.zeros_like(proj_weight, dtype=dtype)

        if reduction == "mean":
            divisor = (target != ignore_index).sum().to(torch.float32)
        else:
            divisor = torch.ones(1, dtype=torch.float32, device=in_feat.device)

        # Divide the input into chunks of size num_tokens // n_loop_iters, then compute the loss for each of these groups
        proj_weight_cast = proj_weight.to(dtype)

        loop_chunk_size = triton.cdiv(n_tokens, n_loop_iters)
        logits_chunk = torch.empty(
            (loop_chunk_size, n_classes), dtype=dtype, device=in_feat.device
        )
        for i, in_feat_chunk in enumerate(torch.split(in_feat, loop_chunk_size)):
            token_start_idx = i * loop_chunk_size
            token_end_idx = (i + 1) * loop_chunk_size

            in_feat_chunk = in_feat_chunk.to(dtype)

            # Compute logits
            torch.matmul(in_feat_chunk, proj_weight_cast.T, out=logits_chunk)

            # Compute loss
            loss_chunk = loss[token_start_idx:token_end_idx]
            targ_chunk = target[token_start_idx:token_end_idx]

            grad_logits_chunk = (
                logits_chunk  # NOTE: we override the logits with their gradients
            )

            cross_entropy_fwd_out(
                x=logits_chunk,
                target=targ_chunk,
                target_logit=None,
                loss=loss_chunk,
                lse=None,
                dx=grad_logits_chunk,
                ignore_index=ignore_index,
            )

            if reduction == "mean":
                loss_chunk /= divisor
                grad_logits_chunk /= divisor

            if in_feat.requires_grad:
                grad_in_feat[token_start_idx:token_end_idx] = (
                    grad_logits_chunk @ proj_weight_cast
                )

            if proj_weight.requires_grad:
                if grad_proj_weight.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        in_feat_chunk, grad_logits_chunk, grad_proj_weight
                    )
                elif grad_proj_weight.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        in_feat_chunk, grad_logits_chunk, grad_proj_weight
                    )
                else:
                    raise RuntimeError(
                        "Unsupported gradient type for gradient accumulation fusion"
                    )

        loss = loss.sum()

        # Save data for backward
        ctx.in_feat_requires_grad = in_feat.requires_grad
        ctx.proj_weight_requires_grad = proj_weight.requires_grad

        if proj_weight.requires_grad and in_feat.requires_grad:
            ctx.save_for_backward(grad_in_feat, grad_proj_weight)
        elif proj_weight.requires_grad and not in_feat.requires_grad:
            ctx.save_for_backward(grad_proj_weight)
        elif not proj_weight.requires_grad and in_feat.requires_grad:
            ctx.save_for_backward(grad_in_feat)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
            grad_in_feat, grad_proj_weight = ctx.saved_tensors
        elif not ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
            (grad_proj_weight,) = ctx.saved_tensors
            grad_in_feat = None
        elif ctx.in_feat_requires_grad and not ctx.proj_weight_requires_grad:
            (grad_in_feat,) = ctx.saved_tensors
            grad_proj_weight = None
        else:
            grad_in_feat = None
            grad_proj_weight = None

        assert grad_output.shape == tuple(), grad_output.shape

        if ctx.in_feat_requires_grad:
            grad_in_feat *= grad_output

        if ctx.proj_weight_requires_grad:
            grad_proj_weight *= grad_output

        return grad_in_feat, grad_proj_weight, None, None, None, None, None, None


def chunked_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    grad_weight: torch.Tensor = None,
    n_loop_iters: int = 1,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    return ChunkedLinearCrossEntropyLossFunction.apply(
        x,
        weight,
        target,
        grad_weight,
        n_loop_iters,
        ignore_index,
        reduction,
    )
