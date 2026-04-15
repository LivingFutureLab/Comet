#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/11 22:20:15
# Author: Shilei Liu
import os
import re
from typing import Optional

from talos.io.filesystem import get_fs_and_access_path
from talos.io.mos import get_latest_uri, is_leaf_uri, is_mos_uri
from talos.io.paths import TRAIN_STATE_NAME


__all__ = [
    "is_leaf_ckpt_dir",
    "check_leaf_dir",
    "find_latest_ckpt",
    "is_valid_ckpt_path",
]


def is_leaf_ckpt_dir(path: str) -> bool:
    fs, path = get_fs_and_access_path(path)
    pattern = r"/checkpoint-\d+/?$"
    return re.search(pattern, path) is not None and fs.exists(path)


def check_leaf_dir(dir: str) -> bool:
    fs, dir = get_fs_and_access_path(dir)
    filename = os.path.join(dir, TRAIN_STATE_NAME)
    return fs.exists(filename)


def find_latest_ckpt(dir: str) -> Optional[str]:
    fs, dir = get_fs_and_access_path(dir)
    if not fs.exists(dir):
        return None

    pattern = re.compile(r".*/checkpoint-\d+$")
    subdirs = []
    for entry in fs.listdir(dir, detail=False):
        entry = entry.rstrip("/")
        if fs.isdir(entry) and pattern.match(entry):
            subdirs.append(entry)

    if len(subdirs) == 0:
        return None

    subdirs.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    target = None
    for subdir in subdirs:
        if check_leaf_dir(subdir):
            target = subdir
            break
    return target


def is_valid_ckpt_path(uri_or_dir: str) -> bool:
    if is_mos_uri(uri_or_dir):
        if is_leaf_uri(uri_or_dir):
            return True
        else:
            return get_latest_uri(uri_or_dir) is not None
    else:
        if is_leaf_ckpt_dir(uri_or_dir):
            return check_leaf_dir(uri_or_dir)
        else:
            return find_latest_ckpt(uri_or_dir) is not None
