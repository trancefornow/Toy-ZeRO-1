"""CUDA memory reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class MemorySnapshot:
    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int


def format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def reset_peak_memory(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(device: Optional[torch.device] = None, synchronize: bool = True) -> MemorySnapshot:
    if not torch.cuda.is_available():
        return MemorySnapshot(allocated=0, reserved=0, max_allocated=0, max_reserved=0)

    if synchronize:
        torch.cuda.synchronize(device)

    return MemorySnapshot(
        allocated=torch.cuda.memory_allocated(device),
        reserved=torch.cuda.memory_reserved(device),
        max_allocated=torch.cuda.max_memory_allocated(device),
        max_reserved=torch.cuda.max_memory_reserved(device),
    )


def format_memory_snapshot(snapshot: MemorySnapshot) -> str:
    return (
        f"allocated={format_bytes(snapshot.allocated)}, "
        f"reserved={format_bytes(snapshot.reserved)}, "
        f"max_allocated={format_bytes(snapshot.max_allocated)}, "
        f"max_reserved={format_bytes(snapshot.max_reserved)}"
    )


def log_memory(prefix: str, device: Optional[torch.device] = None, rank: int = 0) -> None:
    snapshot = memory_snapshot(device)
    print(f"[rank {rank}] {prefix}: {format_memory_snapshot(snapshot)}")

