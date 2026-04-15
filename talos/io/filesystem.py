#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/11 17:11:36
# Author: Shilei Liu
import re
from typing import Tuple
from urllib.parse import urlparse

from fsspec import AbstractFileSystem
from fsspec.implementations.arrow import HadoopFileSystem
from fsspec.implementations.local import LocalFileSystem


__all__ = ["get_filesystem", "get_fs_and_access_path"]


def get_filesystem(path: str) -> AbstractFileSystem:
    prefix = ""
    index = path.find("://")
    if index >= 0:
        prefix = path[:index]
        if prefix == "dfs":
            from pangudfs_client.high_level_client.extern.fsspec import PanguDfs

            fs = PanguDfs()
        elif prefix == "oss":
            from ossfs import OSSFile, OSSFileSystem

            def _upload_chunk_monkey_patch(f, final: bool = False) -> bool:
                f.loc = f.fs.append_object(f.path, f.offset, f.buffer.getvalue())
                return True

            OSSFile._upload_chunk = _upload_chunk_monkey_patch

            _, _, endpoint, access_id, access_key = parse_oss_path(path)
            fs = OSSFileSystem(endpoint=endpoint, key=access_id, secret=access_key)
        elif prefix == "hdfs":
            user, host, port, _ = parse_hdfs_path(path)
            fs = HadoopFileSystem(host=host, port=port, user=user)
        else:
            raise NotImplementedError(f"Does not support `{prefix}` file system.")
    else:
        fs = LocalFileSystem()
    return fs


def parse_oss_path(oss_path: str) -> Tuple[str, str, str, str, str]:
    if re.match(r"^oss://[^:]+:[^@]+@.+$", oss_path) is not None:
        # aliyun odps style.
        bucket, path, endpoint, ak, sk = parse_oss_path_odps_style(oss_path)
    else:
        bucket, path, endpoint, ak, sk = parse_oss_path_http_style(oss_path)
    return bucket, path, endpoint, ak, sk


def parse_oss_path_odps_style(oss_path: str) -> Tuple[str, str, str, str, str]:
    pattern = r"^oss://([^:@]+):([^@]+)@([^/]+)/(.+)$"
    match = re.match(pattern, oss_path, re.IGNORECASE)
    if not match:
        raise ValueError("Invalid OSS URL format")

    ak, sk, endpoint, path = match.groups()
    bucket_name = path.split("/")[0]
    path = path[len(bucket_name) + 1 :]

    return bucket_name, path, endpoint, ak, sk


def parse_oss_path_http_style(oss_path: str) -> Tuple[str, str, str, str, str]:
    from urllib.parse import parse_qs, urlparse

    http_url = oss_path.replace("oss://", "http://", 1)
    parsed_url = urlparse(http_url)
    bucket_name = parsed_url.netloc
    path = parsed_url.path.strip("/")
    query_params = parse_qs(parsed_url.query)
    oss_endpoint = query_params.get("OSS_ENDPOINT", [None])[0]
    oss_access_id = query_params.get("OSS_ACCESS_ID", [None])[0]
    oss_access_key = query_params.get("OSS_ACCESS_KEY", [None])[0]
    return bucket_name, path, oss_endpoint, oss_access_id, oss_access_key


def parse_hdfs_path(hdfs_path: str) -> Tuple[str, str, int, str]:
    parsed = urlparse(hdfs_path)

    user = parsed.username
    host = parsed.hostname
    port = parsed.port
    path = parsed.path
    return user, host, port, path


def get_fs_and_access_path(path: str) -> Tuple[AbstractFileSystem, str]:
    from ossfs import OSSFileSystem

    fs = get_filesystem(path)
    if isinstance(fs, OSSFileSystem):
        bucket_name, path_in_bucket, *_ = parse_oss_path(path)
        access_path = f"oss://{bucket_name}/{path_in_bucket}"
    elif isinstance(fs, HadoopFileSystem):
        *_, access_path = parse_hdfs_path(path)
    else:
        access_path = path

    return fs, access_path
