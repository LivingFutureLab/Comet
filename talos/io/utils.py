#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/08/24 02:08:57
# Author: Shilei Liu
from fnmatch import fnmatch
from typing import List, Optional


__all__ = ["filter_filenames"]


def filter_filenames(
    items: List[str],
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
):
    results = []
    for item in items:
        if allow_patterns is not None and not any(
            fnmatch(item, r) for r in allow_patterns
        ):
            continue

        if ignore_patterns is not None and any(
            fnmatch(item, r) for r in ignore_patterns
        ):
            continue

        results.append(item)
    return results
