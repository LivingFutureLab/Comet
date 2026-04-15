#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/08/20 11:35:08
# Author: Shilei Liu
import json
import math
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from talos.args import HfArgumentParser
from talos.hf.chat_template import SIMPLE_CHAT_TEMPLATE


@dataclass
class Arguments:
    tokenizer_path: str = field(default="/tmp/Qwen/Qwen3-0.6B")
    data_file: str = field(default="/tmp/datas/scolls_32k_chatml_01.jsonl")
    save_dir: str = field(default="/tmp/comet_data/20251202_mixed_scrolls_32k")
    seed: int = field(default=0)
    max_length: int = field(default=32768)
    num_proc: int = field(default=32)


class ChatmlConverter:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, sample):
        messages = json.loads(sample["messages"])
        encoded = self.tokenizer.apply_chat_template(
            messages,
            chat_template=SIMPLE_CHAT_TEMPLATE,
            tokenize=True,
            truncation=True,
            max_length=self.max_len,
            return_dict=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
        input_ids = encoded["input_ids"]
        loss_mask = encoded["assistant_masks"]
        labels = [tid if m == 1 else -100 for tid, m in zip(input_ids, loss_mask)]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class Packer:
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, batch: dict[str, List[int]]) -> dict[str, List[int]]:
        batch_input_ids, batch_labels = batch["input_ids"], batch["labels"]
        packed_input_ids, packed_labels, packed_seq_indices = [], [], []
        input_ids_buffer, labels_buffer, seq_indices_buffer = [], [], []
        seq_idx = 1

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            assert len(input_ids) == len(labels)

            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            seq_len = len(input_ids)

            if sum(labels) == -100 * seq_len:
                continue

            if len(input_ids_buffer) + seq_len > self.max_length:
                packed_input_ids.append(input_ids_buffer)
                packed_labels.append(labels_buffer)
                packed_seq_indices.append(seq_indices_buffer)
                input_ids_buffer, labels_buffer, seq_indices_buffer = [], [], []
                seq_idx = 1

            input_ids_buffer.extend(input_ids)
            labels_buffer.extend(labels)
            seq_indices_buffer.extend([seq_idx] * seq_len)
            seq_idx += 1

        packed_input_ids.append(input_ids_buffer)
        packed_labels.append(labels_buffer)
        packed_seq_indices.append(seq_indices_buffer)

        return {
            "input_ids": packed_input_ids,
            "labels": packed_labels,
            "seq_indices": packed_seq_indices,
        }


def length_statistics(dataset):
    lengths = np.array([len(t) for t in dataset["input_ids"]])
    max_len, minv_len, avgv_len = lengths.max(), lengths.min(), lengths.mean()
    p50, p80, p90, p95, p99 = np.quantile(lengths, [0.5, 0.8, 0.9, 0.95, 0.99]).tolist()
    info = {
        "max": max_len,
        "min": minv_len,
        "avg": avgv_len,
        "p50": p50,
        "p80": p80,
        "p90": p90,
        "p95": p95,
        "p99": p99,
    }
    info = {k: round(v) for k, v in info.items()}
    return info


def save_dataset(dataset, output_dir: str):
    chunk_size = 2048
    data_dir = os.path.join(output_dir, "datas")
    os.makedirs(data_dir, exist_ok=True)

    total_samples = len(dataset)
    num_chunks = math.ceil(total_samples / chunk_size)

    file_list_info = []

    table = dataset.data.table

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)

        chunk = table.slice(start_idx, end_idx - start_idx)

        file_name = f"{i + 1:04d}.parquet"
        relative_path = os.path.join("datas", file_name)
        full_path = os.path.join(output_dir, relative_path)

        pq.write_table(chunk, full_path)

        num_rows = len(chunk)
        file_list_info.append(f"{relative_path} {num_rows}")
        print(f"Saved {file_name} with {num_rows} rows.")

    meta = {"num_chunks": num_chunks, "num_samples": total_samples}
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    with open(os.path.join(output_dir, "filenames.txt"), "w") as f:
        f.write("\n".join(file_list_info))

    print(f"Done! Saved {num_chunks} chunks to {output_dir}")


def main():
    parser = HfArgumentParser(Arguments)
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    print(args)

    tokenizer_path = args.tokenizer_path
    data_file = args.data_file
    seed = args.seed
    save_dir = args.save_dir
    max_length = args.max_length
    num_proc = args.num_proc

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load .jsonl dataset.
    dataset = load_dataset("json", data_files={"train": data_file})
    dataset = dataset["train"]
    print(f"Number of samples before processing: {len(dataset)}")

    # Tokenize and convert to chatml format.
    dataset = dataset.map(
        ChatmlConverter(tokenizer, max_length),
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) >= 4, num_proc=num_proc)
    print(f"Length statistics before packing: {length_statistics(dataset)}")

    # Shuffle the order before packing.
    dataset = dataset.shuffle(seed=seed)

    # Packing.
    dataset = dataset.map(
        Packer(max_length),
        batched=True,
        batch_size=1024,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Packing",
    )
    print(f"Number of samples after packing: {len(dataset)}")
    print(f"Length statistics after packing: {length_statistics(dataset)}")

    save_dataset(dataset, save_dir)


if __name__ == "__main__":
    main()
