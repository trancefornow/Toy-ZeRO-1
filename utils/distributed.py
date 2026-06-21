"""Distributed runtime helpers for torchrun-based training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Runtime information for the current training process."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    distributed: bool

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}.") from exc


def is_torchrun_process() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed(backend: str = "nccl") -> DistributedContext:
    """Initialize torch.distributed when launched by torchrun.

    In single-process runs this function does not initialize a process group.
    It still returns a valid device so scripts can be debugged without torchrun.
    """

    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    distributed = is_torchrun_process() and world_size > 1

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if backend == "nccl" and distributed:
            raise RuntimeError("NCCL distributed training requires CUDA, but torch.cuda.is_available() is False.")
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        distributed=distributed,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def rank_zero_print(*values: object, force: bool = False, **kwargs: object) -> None:
    if force or is_main_process():
        print(*values, **kwargs)


def synchronize_cuda(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

