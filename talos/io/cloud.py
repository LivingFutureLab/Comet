#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/07/09 23:16:50
# Author: Shilei Liu
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import torch.distributed as dist

from talos.dist import broadcast_string, is_local_first_process
from talos.io.filesystem import LocalFileSystem, get_fs_and_access_path
from talos.io.mos import (
    OPENLM_SCHEMA,
    MosCkptFileManager,
    PushToMosResult,
    is_mos_uri,
    push_to_mos,
    repo_download,
)
from talos.io.utils import filter_filenames
from talos.utils.logger import get_logger


__all__ = ["upload_directory", "download_directory", "push", "pull", "CloudFileManager"]

logger = get_logger()


def upload_directory(
    src_dir: str,
    dst_dir: str,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 8,
):
    """Upload files to remote.

    Args:
        src_dir (str): Local directory.
        dst_dir (str): Remote directory.
        max_workers (int, optional): max number of workers. Defaults to 8.

    Raises:
        ValueError: Invalid dir.
    """
    src_dir = Path(src_dir).resolve()
    if not src_dir.is_dir():
        raise ValueError(f"{src_dir} is invalid or not a folder.")

    files = [p for p in src_dir.rglob("*") if p.is_file()]

    if len(files) == 0:
        return

    dst_dir = dst_dir.rstrip("/")
    fs, dst_dir = get_fs_and_access_path(dst_dir)

    remote_dirs = set()
    pairs = []

    for local_file in files:
        rel_path = local_file.relative_to(src_dir).as_posix()
        remote_path = Path(dst_dir) / rel_path
        local_file = str(local_file)
        if len(filter_filenames([local_file], allow_patterns, ignore_patterns)) > 0:
            pairs.append((local_file, str(remote_path)))
            remote_dirs.add(remote_path.parent)

    if isinstance(fs, LocalFileSystem):
        for parent_dir in remote_dirs:
            parent_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(pair: Tuple[str, str]):
        local_path, remote_path = pair
        fs.put(local_path, remote_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(upload_file, pairs)


def download_directory(
    src_dir: str,
    dst_dir: str,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 8,
):
    """Downoad files from remote.

    Args:
        src_dir (str): Remote directory.
        dst_dir (str): Local directory.
        max_workers (int, optional): max number of workers. Defaults to 8.
    """
    dst_dir = Path(dst_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    fs, src_dir = get_fs_and_access_path(src_dir)

    src_dir = src_dir.rstrip("/") + "/"
    files = fs.find(src_dir, detail=False)
    files = filter_filenames(files, allow_patterns, ignore_patterns)

    files_to_show = files if len(files) < 50 else files[:50] + ["..."]
    logger.info(
        f"Downloading the following {len(files)} files to {dst_dir}: {files_to_show}"
    )

    def download_file(remote_path):
        rel_path = remote_path[len(src_dir) :]
        local_path = dst_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_path, str(local_path))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(download_file, files))


def push(
    local_dir: str,
    checkpoint_id: str,
    remote_uri: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 16,
    register_ckpt: bool = False,
) -> Optional[PushToMosResult]:
    """Uploads a local directory to a remote storage location.

    If the remote URI uses the MOS protocol, the upload is handled by `push_to_mos`.
    Otherwise, a generic directory upload is performed to the resolved remote path.

    Args:
        local_dir (str): Path to the local directory to upload.
        checkpoint_id (str): Identifier for the checkpoint (used in remote path or registration).
        remote_uri (Optional[str]): Remote base URI. If MOS, uses specialized uploader; otherwise treated as file path.
        allow_patterns (Optional[List[str]]): File patterns to include (e.g., ["*.bin", "config.json"]).
        ignore_patterns (Optional[List[str]]): File patterns to exclude (e.g., ["*.tmp", "*.log"]).
        max_workers (int): Number of concurrent threads for uploading files. Default: 16.
        register_ckpt (bool): If True and using MOS, registers the checkpoint in the metadata system. Default: False.

    Returns:
        Optional[PushToMosResult]: Metadata about the upload (e.g., version, URI) if MOS is used; otherwise None.

    Notes:
        - For non-MOS remotes, only basic file upload is performed (no registration or versioning).
        - The final remote path is `{remote_uri}/{checkpoint_id}` for non-MOS destinations.
    """
    if is_mos_uri(remote_uri):
        return push_to_mos(
            local_dir,
            checkpoint_id,
            mos_version_uri=remote_uri,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_worker=max_workers,
            register_ckpt=register_ckpt,
        )
    assert remote_uri is not None
    remote_dir = str(Path(remote_uri) / checkpoint_id)
    upload_directory(
        local_dir,
        remote_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers,
    )


def pull(
    remote_uri: str,
    local_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 8,
    once: bool = False,  # Only local rank 0 downloads; others wait.
) -> str:
    """Downloads files from a remote URI to a local directory.

    If once is True, only the local rank 0 process downloads; others wait until it finishes.
    This avoids redundant downloads in distributed training. All processes return the same local_dir.

    Args:
        remote_uri (str): Remote URI (e.g., MOS or file path).
        local_dir (Optional[str]): Local directory to download to.
        allow_patterns (Optional[List[str]]): File patterns to include (e.g., ["*.bin"]).
        ignore_patterns (Optional[List[str]]): File patterns to exclude (e.g., ["*.pt"]).
        max_workers (int): Number of parallel download workers. Default: 8.
        once (bool): If True, only the local rank 0 process performs the download;
            all other processes wait until it finishes. If False, every process downloads
            independently (may cause redundancy). Default: False.

    Returns:
        str: Local directory path where files are downloaded.

    Note:
        When once=True and distributed training is active, a barrier ensures synchronization.
    """
    result = local_dir
    if not once or is_local_first_process():
        if is_mos_uri(remote_uri):
            if remote_uri.startswith(OPENLM_SCHEMA) and "&ckpt" not in remote_uri:
                remote_uri = remote_uri[len(OPENLM_SCHEMA) :]
            result = repo_download(
                remote_uri,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=max_workers,
            )
        else:
            assert local_dir is not None, (
                "`local_dir` must be specified for non-MOS downloads"
            )
            download_directory(
                remote_uri,
                local_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=max_workers,
            )

    if once and dist.is_initialized():
        if local_dir is None:
            result = broadcast_string(result, 0)
        else:
            dist.barrier()

    return result


class CloudFileManager:
    def __init__(self, path: str, mode: str = "r", create_version: bool = False):
        self.path = path
        if is_mos_uri(path):
            self.mgr = MosCkptFileManager(path, mode, create_version)
        else:
            self.mgr = None

    def get_fs_and_access_path(self):
        if self.mgr is None:
            fs, access_path = get_fs_and_access_path(self.path)
        else:
            fs = self.mgr.get_fs()
            access_path = self.mgr.path
        return fs, access_path

    def register_ckpt(
        self,
        metrics: dict = None,
        labels: List[str] = None,  # labels: List[str] = ["key1=value1", "key2=value2"]
    ):
        if self.mgr is not None:
            self.mgr.register_ckpt(metrics=metrics, labels=labels)
