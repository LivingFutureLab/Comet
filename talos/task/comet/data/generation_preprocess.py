#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/09/02 21:27:13
# Author: Shilei Liu
import json
import math
from typing import List, Tuple, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from talos.task.comet.data.chunk_strategy import (
    FixedSizeStrategy,
    RightAlignedFixedSizeStrategy,
)
from talos.task.comet.data.input_builder import InputBuilder
from talos.task.comet.data.train_preprocess import TrainBatchCollator


class Gentransform:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_input_len: int,
        placeholder_id: int,
        chunk_size: int,
        temp_beacon_id: int,
        temp_beacon_stride: int,
        temp_mem_budget: int,
        global_beacon_ids: List[int],
        is_chatml: bool = False,
        is_pre_tokenized: bool = False,
        max_priming_length: int = 0,
    ):
        """Initializes the preprocessor with a tokenizer and configuration options."""
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.is_chatml = is_chatml
        self.is_pre_tokenized = is_pre_tokenized

        self.ph_id = placeholder_id
        self.chunk_size = chunk_size
        self.temp_beacon_id = temp_beacon_id
        self.temp_beacon_stride = temp_beacon_stride
        self.temp_mem_budget = temp_mem_budget
        self.global_beacon_ids = global_beacon_ids
        self.num_beacons = max(math.ceil(chunk_size / temp_beacon_stride), 0)
        self.max_priming_length = max_priming_length

    @classmethod
    def from_config(
        cls,
        config,
        tokenizer: PreTrainedTokenizerBase,
        max_input_len: int,
        is_chatml: bool = False,
        is_pre_tokenized: bool = False,
        max_priming_length: int = 0,
    ):
        return cls(
            tokenizer,
            max_input_len,
            config.placeholder_id,
            config.chunk_size,
            config.temp_beacon_id,
            config.temp_beacon_stride,
            config.temp_mem_budget,
            config.global_beacon_ids,
            is_chatml,
            is_pre_tokenized,
            max_priming_length,
        )

    def tokenize(self, line: Tuple[str, Union[str, List[int]]]):
        uniq_id = line[0]
        if self.is_chatml:
            messages = json.loads(line[1])
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=self.max_input_len,
            )
        elif not self.is_pre_tokenized:
            input_ids = self.tokenizer.encode(
                line[1], truncation=True, max_length=self.max_input_len
            )
        else:
            input_ids = line[1][-self.max_input_len :]
        return uniq_id, input_ids

    def __call__(self, line: Tuple[str, Union[str, List[int]]]):
        uniq_id, input_ids = self.tokenize(line)
        seq_len = len(input_ids)
        if self.max_priming_length > 0 and seq_len > self.chunk_size:
            chunk_sizes = RightAlignedFixedSizeStrategy.build(
                seq_len, self.chunk_size, self.max_priming_length
            )
        else:
            chunk_sizes = FixedSizeStrategy.build(seq_len, self.chunk_size)
        builder = InputBuilder.init(chunk_sizes, input_ids)
        builder.inject_temp_memory_tokens(
            self.temp_beacon_id,
            self.temp_beacon_stride,
            self.temp_mem_budget,
            self.ph_id,
            is_generating=True,
        )
        builder.inject_global_memory_tokens(
            self.global_beacon_ids, self.ph_id, is_generating=True
        )
        builder.init_position_ids()
        seq = builder.fetch()
        return uniq_id, seq


class GenBatchCollator(TrainBatchCollator):
    def __call__(self, examples):
        if len(examples) > 1:
            raise NotImplementedError("Only support batch size equal to 1.")
        uniq_ids = [e[0] for e in examples]
        seqs = [e[1] for e in examples]
        batch = super().__call__(seqs)
        batch.pop("labels")
        last_chunk_size = batch["chunk_sizes"][-1]
        last_chunk_num_temp_beacons = batch["temp_beacon_inds"][-1].shape[1]
        last_chunk_text_length = last_chunk_size - last_chunk_num_temp_beacons
        batch["last_chunk_text_length"] = last_chunk_text_length
        return uniq_ids, batch
