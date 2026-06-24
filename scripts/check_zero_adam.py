"""Compare the single-GPU ZeroAdam update against torch.optim.Adam."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import MLPConfig, build_mlp
from optimizer import ZeroAdam
from utils.data import SyntheticClassificationConfig, make_synthetic_classification
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check ZeroAdam against the single-tensor PyTorch Adam path."
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def compare_parameters(
    reference_model: nn.Module,
    zero_model: nn.Module,
    *,
    atol: float,
    rtol: float,
) -> float:
    max_difference = 0.0
    for (reference_name, reference_param), (zero_name, zero_param) in zip(
        reference_model.named_parameters(),
        zero_model.named_parameters(),
    ):
        if reference_name != zero_name:
            raise RuntimeError(
                f"Parameter order mismatch: {reference_name!r} != {zero_name!r}."
            )
        difference = (reference_param - zero_param).abs().max().item()
        max_difference = max(max_difference, difference)
        torch.testing.assert_close(
            zero_param,
            reference_param,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{reference_name} differs after ZeroAdam step.\n{message}",
        )
    return max_difference


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("ZeroAdam comparison requires the project's single CUDA GPU.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    model_config = MLPConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        num_hidden_layers=args.num_hidden_layers,
        dropout=0.0,
    )
    reference_model = build_mlp(model_config).to(device)
    zero_model = copy.deepcopy(reference_model)

    reference_optimizer = torch.optim.Adam(
        reference_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
    )
    zero_optimizer = ZeroAdam(
        zero_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataset = make_synthetic_classification(
        SyntheticClassificationConfig(
            num_samples=max(args.steps * args.batch_size, args.batch_size),
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            seed=args.seed,
        )
    )
    features, targets = dataset.tensors
    criterion = nn.CrossEntropyLoss()

    max_parameter_difference = 0.0
    for step in range(args.steps):
        start = step * args.batch_size
        end = start + args.batch_size
        inputs = features[start:end].to(device)
        labels = targets[start:end].to(device)

        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = criterion(reference_model(inputs), labels)
        reference_loss.backward()
        reference_optimizer.step()

        zero_optimizer.zero_grad(set_to_none=True)
        zero_loss = criterion(zero_model(inputs), labels)
        zero_loss.backward()
        zero_optimizer.step()

        torch.testing.assert_close(
            zero_loss,
            reference_loss,
            atol=args.atol,
            rtol=args.rtol,
        )
        step_difference = compare_parameters(
            reference_model,
            zero_model,
            atol=args.atol,
            rtol=args.rtol,
        )
        max_parameter_difference = max(max_parameter_difference, step_difference)
        print(
            f"step={step + 1:02d} "
            f"loss={zero_loss.item():.8f} "
            f"max_param_diff={step_difference:.3e}"
        )

    shard_start, shard_end = zero_optimizer.local_shard_range
    if (shard_start, shard_end) != (0, zero_optimizer.total_numel):
        raise RuntimeError("Single-rank local shard does not cover the full parameter vector.")
    if zero_optimizer.local_state_numel != 2 * zero_optimizer.total_numel:
        raise RuntimeError("ZeroAdam state size does not match two flat Adam state tensors.")

    state_owner = zero_optimizer.param_groups[0]["params"][0]
    flat_state = zero_optimizer.state[state_owner]
    if flat_state["step"] != args.steps:
        raise RuntimeError(
            f"ZeroAdam step counter is {flat_state['step']}, expected {args.steps}."
        )
    for state_name in ("exp_avg", "exp_avg_sq"):
        state_tensor = flat_state[state_name]
        if state_tensor.shape != (zero_optimizer.total_numel,):
            raise RuntimeError(
                f"{state_name} has shape {tuple(state_tensor.shape)}, "
                f"expected ({zero_optimizer.total_numel},)."
            )

    print(
        "ZeroAdam comparison passed: "
        f"parameters={zero_optimizer.total_numel:,}, "
        f"local_shard=[{shard_start}, {shard_end}), "
        f"state_elements={zero_optimizer.local_state_numel:,}, "
        f"max_param_diff={max_parameter_difference:.3e}"
    )


if __name__ == "__main__":
    main()
