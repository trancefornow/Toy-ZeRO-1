"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, rank: int = 0) -> int:
    """Seed Python, NumPy, and PyTorch.

    The returned seed includes the optional rank offset. Use rank=0 when every
    process should generate identical synthetic data and model initialization.
    Use rank=ctx.rank when per-rank randomness is required.
    """

    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    return effective_seed

