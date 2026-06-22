"""Smoke test for synthetic data and the MLP model."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import MLPConfig, build_mlp, count_parameters
from utils.data import SyntheticClassificationConfig, make_data_loader, make_synthetic_classification
from utils.distributed import barrier, cleanup_distributed, rank_zero_print, setup_distributed
from utils.memory import format_memory_snapshot, memory_snapshot, reset_peak_memory
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single MLP training smoke step.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def gradient_norm(model: nn.Module, device: torch.device) -> float:
    total = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            total = total + param.grad.detach().float().pow(2).sum()
    return total.sqrt().item()


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()

    try:
        set_seed(args.seed)
        reset_peak_memory(ctx.device)

        data_config = SyntheticClassificationConfig(
            num_samples=args.num_samples,
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            seed=args.seed,
        )
        dataset = make_synthetic_classification(data_config)
        loader = make_data_loader(
            dataset,
            batch_size=args.batch_size,
            rank=ctx.rank,
            world_size=ctx.world_size,
            shuffle=False,
            seed=args.seed,
        )

        model_config = MLPConfig(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.num_classes,
            num_hidden_layers=args.num_hidden_layers,
        )
        model = build_mlp(model_config).to(ctx.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        inputs, targets = next(iter(loader))
        inputs = inputs.to(ctx.device, non_blocking=True)
        targets = targets.to(ctx.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        grad_norm = gradient_norm(model, ctx.device)
        optimizer.step()

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Loss is not finite: {loss.item()}.")
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError(f"Gradient norm is invalid: {grad_norm}.")

        snapshot = memory_snapshot(ctx.device)
        rank_zero_print(
            "Model smoke test passed: "
            f"params={count_parameters(model):,}, "
            f"batch_size={args.batch_size}, "
            f"loss={loss.item():.6f}, "
            f"grad_norm={grad_norm:.6f}, "
            f"memory=({format_memory_snapshot(snapshot)})"
        )
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

