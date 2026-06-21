"""Smoke test for the distributed runtime helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.distributed import barrier, cleanup_distributed, rank_zero_print, setup_distributed
from utils.memory import log_memory, reset_peak_memory
from utils.seed import set_seed


def main() -> None:
    ctx = setup_distributed()
    try:
        set_seed(1234)
        reset_peak_memory(ctx.device)

        rank_zero_print(
            f"distributed={ctx.distributed}, world_size={ctx.world_size}, backend={dist.get_backend() if ctx.distributed else 'none'}"
        )

        value = torch.tensor([float(ctx.rank + 1)], device=ctx.device)
        if ctx.distributed:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

        expected = ctx.world_size * (ctx.world_size + 1) / 2.0
        if not torch.isclose(value.cpu(), torch.tensor([expected])).item():
            raise RuntimeError(f"all_reduce returned {value.item()}, expected {expected}.")

        print(
            f"[rank {ctx.rank}] local_rank={ctx.local_rank}, device={ctx.device}, "
            f"all_reduce_sum={value.item():.1f}"
        )
        log_memory("smoke", device=ctx.device, rank=ctx.rank)
        barrier()
        rank_zero_print("Distributed smoke test passed.")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

