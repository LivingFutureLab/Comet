#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/08 13:04:31
# Author: Shilei Liu
import json
from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from transformers.hf_argparser import HfArgumentParser
from transformers.models.qwen2 import Qwen2TokenizerFast

import talos
from talos.args import DataArguments, GenerationArguments, MixedPrecArguments
from talos.hf.chat_template import SIMPLE_CHAT_TEMPLATE
from talos.task.comet.data.generation_preprocess import GenBatchCollator, Gentransform
from talos.task.comet.modeling_qwen3 import MemQwen3ForCausalLM
from talos.task.comet.utils import to_device


logger = talos.utils.get_logger()


@dataclass
class InferArguments(MixedPrecArguments):
    model_path: str = field(default="/tmp/model")
    tokenizer_path: str = field(default=None)
    max_input_length: int = field(
        default=2048, metadata={"help": "Max sequence length."}
    )
    max_priming_length: int = field(
        default=0,
        metadata={"help": "The length from the end of input to prime generation."},
    )
    eos_token: str = field(default=None)
    skip_special_tokens: bool = field(default=False)
    num_workers: int = field(default=4)
    batch_size: int = field(default=1)

    def __post_init__(self):
        assert self.batch_size == 1


class EvalDataset:
    def __init__(self, path: str, data_convert):
        self.data_convert = data_convert
        self.datas = self.load(path)

    def load(self, path):
        lines = []
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                lines.append((obj["uniq_id"], obj["messages"]))
        return lines

    def __getitem__(self, key):
        return self.data_convert(self.datas[key])

    def __len__(self):
        return len(self.datas)


def get_args() -> Tuple[DataArguments, GenerationArguments, InferArguments]:
    parser = HfArgumentParser((DataArguments, GenerationArguments, InferArguments))
    data_args, gen_args, args = parser.parse_args_into_dataclasses()
    return data_args, gen_args, args


def main():
    data_args, gen_args, args = get_args()
    logger.info(data_args)
    logger.info(gen_args)
    logger.info(args)

    dtype = args.torch_dtype()
    device = torch.device("cuda")

    model_path = args.model_path

    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = model_path

    kwargs = {"device_map": device, "torch_dtype": dtype}

    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path)
    model = MemQwen3ForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()

    tokenizer.truncation_side = "left"
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    config = model.config
    logger.info(f"Config: {config}")
    logger.info(f"#params: {talos.utils.get_num_parameters(model)}")

    if args.eos_token is not None:
        eos_token_id = tokenizer(args.eos_token)["input_ids"][0]
        logger.info(f"Set eos_token_id to {eos_token_id}")
        model.config.eos_token_id = eos_token_id

    func = Gentransform.from_config(
        config,
        tokenizer,
        args.max_input_length,
        is_chatml=True,
        is_pre_tokenized=False,
        max_priming_length=args.max_priming_length,
    )

    dataset = EvalDataset(data_args.tables, func)

    collate_fn = GenBatchCollator()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    generation_config = GenerationConfig(
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        **gen_args.to_dict(),
    )
    model.generation_config = generation_config
    logger.info(f"Generation config: {generation_config}")

    for data in dataloader:
        uniq_ids, batch = data
        batch = to_device(batch, device)
        batch["mem_effi"] = True
        chunk_sizes = batch.get("chunk_sizes")
        talos.utils.tprint(f"Chunk_sizes: {chunk_sizes}")
        input_ids = batch["input_ids"]
        sequences = model.generate(**batch)
        prompt_size = input_ids.shape[1]
        sequences = sequences[:, prompt_size:]
        texts = []
        for i in range(sequences.shape[0]):
            seq = sequences[i]
            valid_seq = seq[seq != config.temp_beacon_id]
            text = tokenizer.decode(
                valid_seq,
                skip_special_tokens=args.skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            texts.append(text.strip())
        talos.utils.tprint(f"Outputs: {texts}")
        assert len(uniq_ids) == len(texts), f"{len(uniq_ids)} v.s. {len(texts)}"


if __name__ == "__main__":
    main()
