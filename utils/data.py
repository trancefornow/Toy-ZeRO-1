"""Synthetic datasets for deterministic training tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


@dataclass(frozen=True)
class SyntheticClassificationConfig:
    num_samples: int = 1024
    input_dim: int = 128
    num_classes: int = 10
    noise_std: float = 0.05
    seed: int = 1234


def make_synthetic_classification(config: SyntheticClassificationConfig) -> TensorDataset:
    """Create a deterministic classification dataset from a random teacher."""

    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if config.input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if config.num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")
    if config.noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    features = torch.randn(config.num_samples, config.input_dim, generator=generator)
    teacher = torch.randn(config.input_dim, config.num_classes, generator=generator)
    logits = features @ teacher
    if config.noise_std > 0.0:
        logits = logits + config.noise_std * torch.randn(
            config.num_samples,
            config.num_classes,
            generator=generator,
        )

    targets = logits.argmax(dim=1).to(torch.long)
    return TensorDataset(features, targets)


def make_data_loader(
    dataset: TensorDataset,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = False,
    seed: int = 1234,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader with an optional DistributedSampler."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )

