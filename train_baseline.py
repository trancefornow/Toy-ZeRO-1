"""Single-GPU Adam baseline for Toy-ZeRO-1."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import MLPConfig, build_mlp, count_parameters
from utils.data import SyntheticClassificationConfig, make_data_loader, make_synthetic_classification
from utils.memory import format_memory_snapshot, memory_snapshot, reset_peak_memory
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-GPU Adam baseline.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
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


def next_batch(
    loader_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    loader: torch.utils.data.DataLoader,
) -> tuple[tuple[torch.Tensor, torch.Tensor], Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("This baseline expects one CUDA GPU, but torch.cuda.is_available() is False.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    reset_peak_memory(device)

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
        rank=0,
        world_size=1,
        shuffle=False,
        seed=args.seed,
    )

    model_config = MLPConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        num_hidden_layers=args.num_hidden_layers,
    )
    model = build_mlp(model_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        "Baseline config: "
        f"params={count_parameters(model):,}, "
        f"steps={args.steps}, "
        f"batch_size={args.batch_size}, "
        f"lr={args.lr}, "
        f"device={device}"
    )

    loader_iter = iter(loader)
    losses: list[float] = []

    model.train()
    for step in range(1, args.steps + 1):
        (inputs, targets), loader_iter = next_batch(loader_iter, loader)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        grad_norm = gradient_norm(model, device)
        optimizer.step()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss became non-finite at step {step}: {loss_value}.")
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Gradient norm became non-finite at step {step}: {grad_norm}.")

        losses.append(loss_value)

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(f"step={step:04d} loss={loss_value:.6f} grad_norm={grad_norm:.6f}")

    snapshot = memory_snapshot(device)
    print(f"Baseline memory: {format_memory_snapshot(snapshot)}")
    print(f"Baseline final_loss={losses[-1]:.6f}")


if __name__ == "__main__":
    main()
