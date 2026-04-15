#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2023/10/06 11:47:34
# Author: Shilei Liu
import itertools
import json
import os
from typing import List, Optional, Tuple, Union

from fsspec import AbstractFileSystem

from talos.data.base import FiniteStreamDataset
from talos.data.utils import (
    cala_local_worker_resume_offset,
    calc_slice_range,
    drop_redundant_data,
)
from talos.io.cloud import CloudFileManager
from talos.utils import get_logger


__all__ = ["ParquetReader", "ChunkFetcher", "ChunkedDataset", "ChunkedDatasets"]


logger = get_logger()

# Define constants for metadata files
META_FILE = "meta.json"
FILENAMES_FILE = "filenames.txt"


class ParquetReader:
    """
    Reads a single Parquet file in a streaming fashion by iterating through its
    row groups. This approach is memory-efficient as it avoids loading the
    entire file into memory at once.

    Args:
        fs (AbstractFileSystem): An fsspec-compatible filesystem instance (e.g., for local, S3, HDFS).
        path (str): The path to the Parquet file.
        cols (List[str]): A list of column names to read from the file.
    """

    def __init__(self, fs: AbstractFileSystem, path: str, cols: List[str]):
        import pyarrow.parquet as pq
        from pyarrow.fs import FSSpecHandler, PyFileSystem

        # Create a PyArrow filesystem from the fsspec filesystem to allow PyArrow to
        # access various storage backends through fsspec.
        arrow_fs = PyFileSystem(FSSpecHandler(fs))

        # Open the Parquet file. This operation is lazy and only reads metadata,
        # not the actual data, making it fast and memory-light.
        self.parquet_file = pq.ParquetFile(arrow_fs.open_input_file(path))
        self.cols = cols
        self.row_group_iterator = range(self.parquet_file.num_row_groups)

    def __iter__(self):
        """
        Yields rows one by one from the Parquet file.
        """
        # Iterate over each row group in the file.
        for i in self.row_group_iterator:
            # `read_row_group` loads only one row group into memory, which is
            # typically much smaller than the entire file.
            table = self.parquet_file.read_row_group(i, columns=self.cols)

            # A table can be composed of multiple record batches.
            for batch in table.to_batches():
                # `zip` the columns to iterate through the data row by row.
                for row_in_columns in zip(*(batch[c] for c in self.cols)):
                    # Convert PyArrow scalar types to native Python types for downstream use.
                    py_row = tuple(item.as_py() for item in row_in_columns)
                    yield py_row


class ChunkFetcher:
    """
    Fetches items sequentially from a collection of Parquet chunk files.

    It manages the process of switching between chunk files and can start reading
    from any global position across the entire dataset. It uses the `ParquetReader`
    to keep memory usage low.

    Args:
        fs (AbstractFileSystem): The filesystem instance.
        data_dir (str): The directory containing the Parquet chunks and metadata files.
        cols (List[str]): The columns to read.
        start_pos (int): The global starting row index for this fetcher.
        end_pos (int): The global ending row index (exclusive) for this fetcher.
    """

    def __init__(
        self,
        fs: AbstractFileSystem,
        data_dir: str,
        cols: List[str],
        start_pos: int,
        end_pos: int,
    ):
        self.fs = fs
        self.data_dir = data_dir
        self.cols = cols
        self.start_pos = start_pos
        self.end_pos = end_pos

        # Load the list of chunk filenames and their corresponding row counts.
        self.filenames, self.chunk_sizes = self._read_all_filenames()

        # Determine the initial chunk index and the starting position within that chunk
        # based on the global `start_pos`.
        self.cur_chunk_idx, self.cur_pos_in_chunk = self._init_data_ptr(
            self.chunk_sizes, start_pos
        )

        self.cur_global_pos = start_pos
        # Load the iterator for the first chunk to be read.
        self.cur_chunk_iterator = self._load_chunk_iterator(self.cur_chunk_idx)

    def _read_all_filenames(self) -> Tuple[List[str], List[int]]:
        """Reads filenames and their corresponding sizes from the FILENAMES_FILE."""
        path = os.path.join(self.data_dir, FILENAMES_FILE)
        filenames, chunk_sizes = [], []
        # The file format is assumed to be "filename.parquet 12345".
        for line in self.fs.read_text(path).split("\n"):
            if line == "":
                continue
            filename, chunk_size = line.split(" ")
            chunk_size = int(chunk_size)
            filenames.append(filename)
            chunk_sizes.append(chunk_size)
        return filenames, chunk_sizes

    @staticmethod
    def _init_data_ptr(chunk_sizes: List[int], start_pos: int) -> Tuple[int, int]:
        """
        Calculates the starting chunk index and the relative position within that chunk
        for a given global `start_pos`.
        """
        accumulated_size = 0
        for i, chunk_size in enumerate(chunk_sizes):
            # Check if the target `start_pos` falls within the current chunk.
            if accumulated_size + chunk_size > start_pos:
                # The index is `i`, and the position inside is the remainder.
                return i, start_pos - accumulated_size
            accumulated_size += chunk_size

        raise ValueError(
            f"`start_pos` {start_pos} is out of bounds for a dataset of "
            f"total size {accumulated_size}."
        )

    def _load_chunk_iterator(self, chunk_idx: int) -> iter:
        """
        Loads a new chunk file and returns an iterator over its rows. It can
        efficiently discard initial rows if starting from a mid-chunk position.
        """
        if chunk_idx >= len(self.filenames):
            # Return an empty iterator if we are past the last chunk.
            return iter([])

        filename = os.path.join(self.data_dir, self.filenames[chunk_idx])

        # Use the memory-efficient ParquetReader for streaming.
        reader = ParquetReader(self.fs, filename, self.cols)
        stream = iter(reader)

        # If we need to start from a position other than the beginning of the chunk,
        # we efficiently discard the items we don't need.
        if self.cur_pos_in_chunk > 0:
            # `itertools.islice` consumes the iterator without loading the
            # skipped elements into memory, which is crucial for performance.
            stream = itertools.islice(stream, self.cur_pos_in_chunk, None)

        return stream

    def fetch(self):
        """
        Fetches the next item from the stream of chunks, automatically handling
        the transition to the next chunk when the current one is exhausted.
        """
        while True:
            if self.cur_global_pos >= self.end_pos:
                raise StopIteration()

            try:
                # Get the next item from the current chunk's iterator.
                item = next(self.cur_chunk_iterator)
                self.cur_global_pos += 1
                return item
            except StopIteration:
                # This block executes when the current chunk is fully consumed.
                self.cur_chunk_idx += 1

                # If there are no more chunks, signal the end of the iteration.
                if self.cur_chunk_idx >= len(self.filenames):
                    raise StopIteration()

                # For any new chunk, we always start reading from its beginning (position 0).
                self.cur_pos_in_chunk = 0
                self.cur_chunk_iterator = self._load_chunk_iterator(self.cur_chunk_idx)


class ChunkedDataset(FiniteStreamDataset):
    """
    A PyTorch-compatible streaming dataset for a single directory of chunked Parquet files.

    This class handles distributed data loading by calculating the specific slice
    of the total data that each distributed process (rank) is responsible for reading.

    Args:
        data_dir (str): Directory containing Parquet chunks and metadata.
        cols (str): Comma-separated string of column names to load.
        drop (bool): If True, drops trailing samples to make the dataset evenly
                     divisible among slices (ranks). This ensures each rank
                     processes the exact same number of samples.
        slice_id (int): The ID (rank) of the current process in distributed training.
        slice_count (int): The total number of processes (world size).
    """

    def __init__(
        self,
        data_dir: str,
        cols: str,
        drop: bool = False,
        slice_id: Optional[int] = None,
        slice_count: Optional[int] = None,
    ):
        super().__init__(slice_id, slice_count)
        self.data_dir = data_dir
        self.drop = drop
        self.cols = cols.split(",")
        self.offset_per_slice = 0  # Offset used for resuming training.

        # Get total dataset info and calculate the data range for this slice.
        num_chunks, num_samples, start, end = self.get_basic_info(
            self.slice_id, self.slice_count
        )

        self.num_row_local = end - start  # Number of samples for this slice
        self.num_row_global = num_samples  # Total number of samples in the dataset
        self.start_pos = start  # Global start index for this slice
        self.end_pos = end  # Global end index for this slice

        # If `drop` is True, adjust counts to ensure even distribution across slices.
        if self.drop and self.num_row_global % self.slice_count != 0:
            diff_global, diff_local = drop_redundant_data(
                self.num_row_global, self.num_row_local, self.slice_count
            )
            self.num_row_global -= diff_global

            logger.info(
                f"{diff_global} samples were discarded to keep "
                "#rows consistent on each worker."
            )
            if diff_local > 0:
                self.num_row_local -= diff_local
                self.end_pos -= diff_local  # Adjust the end position for this slice.
                logger.info(f"drop {diff_local} samples on slice {self.slice_id}.")

        logger.info(
            f"Dataset loaded: {self.num_row_global} total samples ({num_chunks} chunks). "
            f"Slice {self.slice_id}/{self.slice_count} will process {self.num_row_local} samples, "
            f"from global index {self.start_pos} to {self.end_pos}."
        )

    def get_basic_info(
        self, slice_id: int = 0, slice_count: int = 1
    ) -> Tuple[int, int, int, int]:
        """
        Reads metadata from `meta.json` and calculates the data range for the current slice.
        """
        fs, data_dir = CloudFileManager(self.data_dir).get_fs_and_access_path()
        path = os.path.join(data_dir, META_FILE)
        with fs.open(path, "r") as f:
            meta = json.loads(f.read())
        num_chunks: int = meta["num_chunks"]
        num_samples: int = meta["num_samples"]
        # `calc_slice_range` is a utility to divide a total number of items among slices.
        start, end = calc_slice_range(num_samples, slice_id, slice_count)
        return num_chunks, num_samples, start, end

    @property
    def num_samples_local(self) -> int:
        """Number of samples to be processed by the current slice (process/rank)."""
        return self.num_row_local

    @property
    def num_samples_global(self) -> int:
        """Total number of samples in the entire dataset (after dropping, if any)."""
        return self.num_row_global

    def reset_data_ptr(self, consumed_samples: int = 0):
        """
        Sets an offset to skip samples, used for resuming training from a checkpoint.
        """
        if consumed_samples == 0:
            return
        # Ensure the consumed samples wraps around if it exceeds the total dataset size.
        consumed_samples %= self.num_row_global
        # Distribute the total consumed samples among the slices.
        self.offset_per_slice = consumed_samples // self.slice_count
        logger.info(
            f"{self.offset_per_slice} samples will be skipped on this slice to resume."
        )

    def clear_per_slice_offset(self):
        """Resets the resume offset, typically at the start of a new epoch."""
        self.offset_per_slice = 0

    def set_epoch(self, epoch: int):
        """Called at the beginning of each epoch to handle epoch-specific logic."""
        super().set_epoch(epoch)
        # Reset resume offset for the new epoch.
        self.clear_per_slice_offset()

    def build_data_stream(self, worker_id: int, num_workers: int):
        """
        Builds the data stream for a specific DataLoader worker. This method is called
        by each worker process.
        """
        # First, further divide this process's slice of data among its DataLoader workers.
        start_pos, end_pos = calc_slice_range(
            self.num_row_local, worker_id, num_workers, self.start_pos
        )

        # Distribute the per-slice resume offset among the workers of this slice.
        offset_this_worker = cala_local_worker_resume_offset(
            self.offset_per_slice, worker_id, num_workers
        )
        start_pos += offset_this_worker

        # Create a ChunkFetcher for this specific worker, with its calculated start and end positions.
        fs, data_dir = CloudFileManager(self.data_dir).get_fs_and_access_path()
        reader = ChunkFetcher(fs, data_dir, self.cols, start_pos, end_pos)

        # Continuously fetch data until the worker's assigned range is exhausted.
        while True:
            try:
                data = reader.fetch()
            except StopIteration:
                break
            # Apply any mapping function (e.g., tokenization) before yielding.
            yield self._map_func(data)


class ChunkedDatasets(FiniteStreamDataset):
    """
    A meta-dataset that aggregates multiple `ChunkedDataset` instances into a single,
    continuous stream. This is useful for training on multiple datasets sequentially
    as if they were one large dataset.

    Args:
        data_dirs (Union[str, List[str]]): A list of data directories, or a single
                                           comma-separated string of directories.
        cols (str): Comma-separated string of column names to load.
        drop (bool): Whether to drop trailing samples for even distribution.
        slice_id (int): The ID (rank) of the current process.
        slice_count (int): The total number of processes (world size).
    """

    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        cols: str,
        drop: bool = False,
        slice_id: Optional[int] = None,
        slice_count: Optional[int] = None,
    ) -> None:
        super().__init__(slice_id, slice_count)
        if isinstance(data_dirs, str):
            data_dirs = data_dirs.split(",")
        self.data_dirs = data_dirs

        # Create a `ChunkedDataset` for each provided directory.
        self.sub_tabs = [
            ChunkedDataset(table_path, cols, drop, slice_id, slice_count)
            for table_path in data_dirs
        ]
        # Aggregate the sample counts from all sub-datasets.
        self.row_count_global = sum([x.num_samples_global for x in self.sub_tabs])
        self.row_count_local = sum([x.num_samples_local for x in self.sub_tabs])
        self.curr_tab_idx = 0  # Index of the sub-dataset currently being read.

    @property
    def num_samples_local(self) -> int:
        """Total number of samples for this slice across all sub-datasets."""
        return self.row_count_local

    @property
    def num_samples_global(self) -> int:
        """Total number of samples across all sub-datasets."""
        return self.row_count_global

    def _get_current_subtable(self, consumed_samples: int) -> Tuple[int, int]:
        """
        Determines which sub-dataset and what offset within it corresponds to a
        global number of `consumed_samples`. Crucial for resuming.
        """
        accu = 0
        cur_tab = 0
        for i in range(len(self.sub_tabs)):
            summ = accu + self.sub_tabs[i].num_samples_global
            if summ > consumed_samples:
                cur_tab = i
                break
            accu = summ
        # The offset is the number of samples consumed relative to the start of that sub-dataset.
        offset = consumed_samples - accu
        return cur_tab, offset

    def map(self, func):
        """Applies a mapping function to all sub-datasets."""
        for sub in self.sub_tabs:
            sub.map(func)
        return self

    def set_epoch(self, epoch: int):
        """Propagates the set_epoch call to all sub-datasets."""
        super().set_epoch(epoch)
        for sub in self.sub_tabs:
            sub.clear_per_slice_offset()

    def reset_data_ptr(self, consumed_samples: int = 0):
        """
        Resumes from a global sample count by finding the correct sub-dataset
        and setting its internal offset.
        """
        if consumed_samples == 0:
            return

        # Find which sub-dataset and offset to start from.
        cur_tab, offset = self._get_current_subtable(consumed_samples)

        # Set the resume pointer on that specific sub-dataset.
        self.sub_tabs[cur_tab].reset_data_ptr(offset)
        logger.info(f"Resuming from data directory: {self.data_dirs[cur_tab]}")

        # Set the starting sub-dataset index.
        self.curr_tab_idx = cur_tab
        logger.info(
            f"Combined dataset: {len(self.sub_tabs)} tables, "
            f"{self.row_count_global} total samples, "
            f"{self.row_count_local} samples on this slice.",
        )

    def build_data_stream(self, worker_id: int, num_workers: int):
        """
        Builds the combined data stream by iterating through the sub-datasets
        sequentially and yielding from their data streams.
        """
        # Start from the `curr_tab_idx` which may have been set by `reset_data_ptr`.
        for i in range(self.curr_tab_idx, len(self.data_dirs)):
            self.curr_tab_idx = i
            if worker_id == 0:  # Log only from the first worker to avoid spam.
                logger.warning(f"Switching to dataset: {self.data_dirs[i]}.")

            # `yield from` is a clean way to delegate iteration to the sub-generator.
            yield from self.sub_tabs[i].build_data_stream(worker_id, num_workers)
