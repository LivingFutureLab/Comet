#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/04/29 13:16:13
# Author: Shilei Liu
from typing import Optional

from talos.data.chunk import ChunkedDatasets
from talos.data.placeholder import DatasetPlaceholder


__all__ = ["get_dataset"]


def get_dataset(
    impl: str,
    tables: str,
    select_cols: str,
    slice_id: Optional[int] = None,
    slice_count: Optional[int] = None,
    is_placeholder: bool = False,
    drop: bool = True,
):
    """Create a dataset instance based on the specified implementation.

    Args:
        impl: Dataset implementation type. Supported values: "odps", "tunnel_odps", "chunk", "ailake".
        tables: Table names or paths to load data from.
        select_cols: Column names to select from the tables.
        slice_id: ID of the current data slice for distributed training. Defaults to None.
        slice_count: Total number of data slices for distributed training. Defaults to None.
        is_placeholder: If True, returns a placeholder dataset that preserves metadata
            but yields no actual data. Used for non-data-input ranks in model parallelism
            where ranks need dataset metadata (sample counts, slice info) but don't
            process actual training data.
        drop: If True, removes excess data to ensure each worker has the same amount of data.

    Returns:
        Dataset instance configured with the specified parameters.

    Raises:
        KeyError: If the specified impl is not supported.
    """
    if impl == "odps":
        raise NotImplementedError("Please use internal codebase.")
    elif impl == "tunnel_odps":
        raise NotImplementedError("Please use internal codebase.")
    elif impl == "chunk":
        dataset = ChunkedDatasets(
            tables, select_cols, drop=drop, slice_id=slice_id, slice_count=slice_count
        )
    elif impl == "ailake":
        raise NotImplementedError("Please use internal codebase.")
    else:
        raise KeyError(f"Dataset impl `{impl}` is invalid")

    if is_placeholder:
        dataset = DatasetPlaceholder(
            dataset.num_samples_local,
            dataset.num_samples_global,
            dataset.slice_id,
            dataset.slice_count,
        )
    return dataset
