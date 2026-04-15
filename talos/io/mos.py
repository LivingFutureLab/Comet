#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/08/30 18:31:07
# Author: Shilei Liu
# This file has been desensitized.

__all__ = [
    "PushToMosResult",
    "MosCkptFileManager",
    "repo_download",
    "download",
    "push_to_mos",
    "register_ckpt",
    "get_latest_uri",
    "is_leaf_uri",
    "is_mos_uri",
    "OPENLM_SCHEMA",
]

MODEL_CKPT_FLAG = "NONE"
MOS_MODEL_FLAG = "NONE"
MOS_VERSION_FLAG = "NONE"
OPENLM_SCHEMA = "NONE"
PRETRAIN_MODEL_SOURCE_OPENLM_URI_PREFIX = "NONE"


def download(*args, **kwargs):
    pass


def get_last_ckpt_by_uri(*args, **kwargs):
    raise NotImplementedError()


def push_to_mos(*args, **kwargs):
    pass


def register_ckpt(*args, **kwargs):
    pass


def repo_download(*args, **kwargs):
    pass


class MosCkptFileManager:
    pass


class PushToMosResult:
    pass


def get_latest_uri(uri: str):
    try:
        info = get_last_ckpt_by_uri(uri)
        result = info.uri
    except Exception:
        result = None
    return result


def is_leaf_uri(uri: str):
    return "/ckpt_id=" in uri


def is_mos_uri(uri: str) -> bool:
    if uri.startswith(MOS_MODEL_FLAG):
        return True
    if uri.startswith(OPENLM_SCHEMA):
        return True
    if uri.startswith(PRETRAIN_MODEL_SOURCE_OPENLM_URI_PREFIX):
        return True
    if MOS_VERSION_FLAG in uri:
        return True
    if MODEL_CKPT_FLAG in uri:
        return True
    return False
