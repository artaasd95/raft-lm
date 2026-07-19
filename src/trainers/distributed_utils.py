"""Helpers for distributed process-group lifecycle."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch.distributed as dist


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def infer_distributed_env() -> tuple[int, int, int]:
    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)
    return rank, world_size, local_rank


def init_process_group(backend: str = "gloo") -> tuple[int, int, int]:
    rank, world_size, local_rank = infer_distributed_env()
    if world_size <= 1:
        return rank, world_size, local_rank
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this environment")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def distributed_session(backend: str = "gloo") -> Iterator[tuple[int, int, int]]:
    context = init_process_group(backend=backend)
    try:
        yield context
    finally:
        cleanup_process_group()
