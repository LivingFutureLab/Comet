#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/04 20:44:51
# Author: Shilei Liu
# Description: Train CoMeT with Qwen3 as backbone.
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer

import talos
from talos.args import DataArguments, HfArgumentParser, TrainingArguments
from talos.hf.qwen3 import qwen3_train_tflops
from talos.hf.utils import fsdp_wrap_hf_model
from talos.task.comet.data.chunk_plan import DefaultChunkPlanner
from talos.task.comet.data.train_preprocess import TrainBatchCollator, TrainTransform
from talos.task.comet.distributed import (
    MemLayerContextParallelismPlugin,
    ParallelState,
    batch_dispatch,
)
from talos.task.comet.modeling_qwen3 import MemQwen3Config, MemQwen3ForCausalLM
from talos.task.comet.utils import calc_static_chunk_sizes
from talos.train.callback import LogCallback, ProfileCallback, TrackerCallback


logger = talos.utils.get_logger()


@dataclass
class TrainArguments(TrainingArguments):
    config_path: Optional[str] = field(
        default=None, metadata={"help": "Huggingface config path."}
    )
    tokenizer_path: Optional[str] = field(
        default=None, metadata={"help": "Huggingface tokenizer path."}
    )
    remote_model_path: Optional[str] = field(
        default=None, metadata={"help": "Remote pre-trained model path."}
    )
    model_path: str = field(
        default="/tmp/cache/pre_trained", metadata={"help": "Pre-trained model path."}
    )
    max_seq_length: int = field(default=2048, metadata={"help": "Max sequence length."})
    chunk_size: int = field(default=1024)
    global_mem_size: int = field(default=128)
    temp_beacon_stride: int = field(default=64)
    temp_mem_budget: int = field(default=256)
    zero3: bool = field(
        default=False,
        metadata={"help": "Enable ZeRO-3 (otherwise ZeRO-2 will be used)."},
    )
    recompute: bool = field(
        default=False,
        metadata={"help": "Enable activation checkpointing."},
    )
    input_ids_col: str = field(default="input_ids")
    labels_col: str = field(default="labels")
    max_compress_ratio: int = field(default=128)
    dynamic_compress_ratio: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        assert self.max_seq_length % self.chunk_size == 0
        num_chunks = self.max_seq_length // self.chunk_size
        assert self.pp_size >= 1
        assert num_chunks % self.pp_size == 0
        assert self.temp_beacon_stride == -1 or self.temp_beacon_stride > 0


class CometLogCallback(LogCallback):
    def __init__(self, args: TrainArguments, config: MemQwen3Config):
        self.world_size = talos.dist.get_world_size()
        self.config = config
        self.chunk_sizes = calc_static_chunk_sizes(
            args.max_seq_length,
            args.chunk_size,
            len(self.config.global_beacon_ids),
            self.config.temp_beacon_stride,
            self.config.temp_mem_budget,
        )
        logger.info(f"Static chunk sizes: {self.chunk_sizes}")

    def on_log(self, args: TrainArguments, state, logs, **kwargs):
        qps = logs["qps"] / self.world_size
        tps = qps * args.max_seq_length
        # TFLOPs calculation excludes memory access and
        # projection operations (e.g., nn.Linear).
        tflops = sum(qwen3_train_tflops(self.config, c, qps) for c in self.chunk_sizes)
        logs["tps per gpu"] = int(tps)
        logs["tflops"] = int(tflops * 1.3333 if args.recompute else tflops)
        return super().on_log(args, state, logs, **kwargs)


class CometTrainer(talos.train.HFTrainer):
    def __init__(self, parallel_state: ParallelState, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_state = parallel_state

    def compute_loss(self, model, batch):
        if self.parallel_state.world_size > 1:
            batch = batch_dispatch(self.parallel_state, batch)
        batch["use_cache"] = False
        outputs = model(**batch)
        loss = outputs.loss
        return loss

    def create_optimizer(self):
        args = self.args
        grouped_params = self.group_params(self.model, args.weight_decay, args.max_lr)
        optimizer = torch.optim.AdamW(
            grouped_params,
            lr=args.max_lr,
            betas=(args.beta1, args.beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        return optimizer

    @staticmethod
    def group_params(model: torch.nn.Module, weight_decay: float, lr: float):
        decay_params = []
        no_decay_params = []
        decay_mem_params = []
        no_decay_mem_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "global_memory_net" in n or "temp_memory_net" in n:
                (no_decay_mem_params if p.ndim <= 1 else decay_mem_params).append(p)
            else:
                (no_decay_params if p.ndim <= 1 else decay_params).append(p)
        # NOTE: hard code.
        grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": decay_mem_params, "weight_decay": weight_decay, "lr": 5 * lr},
            {"params": no_decay_mem_params, "weight_decay": 0.0, "lr": 5 * lr},
        ]
        logger.info(f"Num decay params: {len(decay_params)}")
        logger.info(f"Num no decay params: {len(no_decay_params)}")
        logger.info(f"Num decay memory params: {len(decay_mem_params)}")
        logger.info(f"Num no decay memory params: {len(no_decay_mem_params)}")
        return grouped_parameters


def get_num_parameters(model: MemQwen3ForCausalLM) -> dict[str, str]:
    """Calculate the total and non-embedding parameter counts
    (where non-embedding excludes the token embeddings and the final lm_head.
    """
    n_params, n_backbone_params, n_trainable = 0, 0, 0
    for n, p in model.named_parameters():
        if model.config.tie_word_embeddings and n.startswith("lm_head"):
            continue
        np = p.numel()
        if not (n.startswith("lm_head") or n.startswith("model.embed_tokens")):
            n_backbone_params += np
        if p.requires_grad:
            n_trainable += np
        n_params += np
    info = {
        "total": f"{n_params:,}",
        "backbone": f"{n_backbone_params:,}",
        "trainable": f"{n_trainable:,}",
    }
    return info


def get_args() -> Tuple[DataArguments, TrainArguments]:
    parser = HfArgumentParser((DataArguments, TrainArguments))
    data_args, args = parser.parse_args_into_dataclasses()
    return data_args, args


def init_model(args: TrainArguments) -> MemQwen3ForCausalLM:
    if args.config_path is not None:
        assert args.tokenizer_path is not None
        # Train from scratch.
        config: MemQwen3Config = MemQwen3Config.from_json_file(args.config_path)
        tokenizer_path = args.tokenizer_path
    else:
        config: MemQwen3Config = MemQwen3Config.from_pretrained(args.model_path)
        if args.tokenizer_path is not None:
            tokenizer_path = args.tokenizer_path
        else:
            tokenizer_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    gmb_new_tokens = [f"<beacon_{i}>" for i in range(args.global_mem_size)]
    tmb_new_tokens = ["<beacon>"]
    placeholder = ["<|memory|>"]

    tokenizer.add_special_tokens({"additional_special_tokens": placeholder})
    tokenizer.add_tokens(gmb_new_tokens + tmb_new_tokens)

    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)[0]
    global_beacon_ids = tokenizer.convert_tokens_to_ids(gmb_new_tokens)
    temp_beacon_id = tokenizer.convert_tokens_to_ids(tmb_new_tokens)[0]

    config.temp_mem_budget = args.temp_mem_budget
    config.temp_beacon_stride = args.temp_beacon_stride
    config.chunk_size = args.chunk_size
    config.placeholder_id = placeholder_id
    config.global_beacon_ids = global_beacon_ids
    config.temp_beacon_id = temp_beacon_id
    config.pad_token_id = config.eos_token_id

    if args.config_path is not None:
        config.vocab_size = len(tokenizer)
        model = MemQwen3ForCausalLM(config)
    else:
        model = MemQwen3ForCausalLM.from_pretrained(
            args.model_path, config=config, torch_dtype=torch.float32
        )
        model.resize_token_embeddings(
            len(tokenizer), pad_to_multiple_of=256, mean_resizing=True
        )

    logger.info(f"Attention implementation: {model.config._attn_implementation}")
    return tokenizer, model


def get_chunk_planner(tokenizer):
    BOUNDARY_TOKENS = ["<|im_end|>", "\n", "\n\n", ".", ",", "。"]
    boundary_token_ids = []
    for token in BOUNDARY_TOKENS:
        ids = tokenizer(token).input_ids
        assert len(ids) == 1
        boundary_token_ids.append(ids[0])
    planner = DefaultChunkPlanner(boundary_token_ids)
    return planner


def main():
    data_args, args = get_args()
    logger.info(data_args)
    logger.info(args)

    talos.init.initialize(args)
    talos.init.setup()

    tokenizer, model = init_model(args)
    config: MemQwen3Config = model.config

    logger.info(f"Config: {config}")
    logger.info(f"Num params: {get_num_parameters(model)}")

    mesh_shape = (args.pp_size, dist.get_world_size() // args.pp_size)
    mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=["pp", "dp"])

    pp_group, dp_group = mesh.get_group("pp"), mesh.get_group("dp")
    parallel_state = ParallelState(pp_group)
    dp_rank, dp_world_size = dist.get_rank(dp_group), dist.get_world_size(dp_group)

    if args.pp_size > 1:
        plugin = MemLayerContextParallelismPlugin(
            config.hidden_size, config.temp_mem_budget, parallel_state
        )
        model.set_plugin(plugin)
        logger.warning(f"Context parallelism group: {mesh['pp'].mesh.tolist()}")
        logger.warning(f"FSDP group: {mesh['dp'].mesh.tolist()}")

    kwargs = {
        "mp_policy": talos.fsdp.get_mp_policy(args.dtype),
        "reshard_after_forward": True if args.zero3 else False,
        "mesh": mesh if args.pp_size > 1 else None,
    }
    fsdp_wrap_hf_model(model, kwargs, args.recompute)
    model.tie_weights()
    logger.info(f"Wrap model done, model arch: {model}")

    planner = get_chunk_planner(tokenizer)
    transform = TrainTransform.from_config(config, args.max_seq_length, planner)
    if config.temp_beacon_stride != -1 and args.dynamic_compress_ratio:
        transform.enable_dynamic_compress_ratio(args.max_compress_ratio)

    dataset = talos.data.get_dataset(
        data_args.data_source,
        data_args.tables,
        f"{args.input_ids_col},{args.labels_col}",
        slice_id=dp_rank,
        slice_count=dp_world_size,
        is_placeholder=not parallel_state.is_first_rank,
    )
    dataset.map(transform)
    logger.info("Initialize dataset done.")

    collate_fn = TrainBatchCollator()

    callbacks = [CometLogCallback(args, config), TrackerCallback(args)]
    callbacks += [ProfileCallback(args.trace_path)] if args.do_profile else []
    trainer = CometTrainer(
        parallel_state,
        args,
        dataset,
        model,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
        callbacks=callbacks,
    )
    trainer.train()

    talos.init.cleanup()


if __name__ == "__main__":
    main()
