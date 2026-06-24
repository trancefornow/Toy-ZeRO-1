"""Single-rank reference implementation of ZeRO-1-style Adam state storage."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


@dataclass(frozen=True)
class FlatParameterSlice:
    """Location of one model parameter inside the flattened parameter vector."""

    start: int
    end: int
    shape: torch.Size

    @property
    def numel(self) -> int:
        return self.end - self.start


class ZeroAdam(Optimizer):
    """Adam with optimizer states stored as one local flat shard.

    This first implementation intentionally targets one process and one GPU.
    With ``world_size=1``, the local shard spans the complete flattened
    parameter vector. Parameters remain owned by the model; only Adam's
    ``exp_avg`` and ``exp_avg_sq`` states are stored in flattened form.

    All trainable parameters must use the same device and dtype, and every
    parameter must receive a dense gradient on each step. These constraints
    keep one shared step counter mathematically equivalent to standard Adam.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}.")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}.")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}.")

        trainable_params = [param for param in params if param.requires_grad]
        if not trainable_params:
            raise ValueError("ZeroAdam requires at least one trainable parameter.")

        device = trainable_params[0].device
        dtype = trainable_params[0].dtype
        if not dtype.is_floating_point:
            raise TypeError(f"ZeroAdam requires floating-point parameters, got {dtype}.")

        for index, param in enumerate(trainable_params):
            if param.device != device:
                raise ValueError(
                    "ZeroAdam requires all parameters on one device, "
                    f"but parameter {index} is on {param.device} instead of {device}."
                )
            if param.dtype != dtype:
                raise ValueError(
                    "ZeroAdam requires all parameters to share one dtype, "
                    f"but parameter {index} has {param.dtype} instead of {dtype}."
                )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(trainable_params, defaults)

        self._parameters = trainable_params
        self._flat_slices = self._build_flat_slices(trainable_params)
        self._total_numel = sum(param.numel() for param in trainable_params)
        self._device = device
        self._dtype = dtype

    @staticmethod
    def _build_flat_slices(parameters: list[Tensor]) -> tuple[FlatParameterSlice, ...]:
        slices: list[FlatParameterSlice] = []
        offset = 0
        for param in parameters:
            end = offset + param.numel()
            slices.append(FlatParameterSlice(start=offset, end=end, shape=param.shape))
            offset = end
        return tuple(slices)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self.param_groups:
            raise NotImplementedError("ZeroAdam currently supports exactly one parameter group.")
        super().add_param_group(param_group)

    @property
    def total_numel(self) -> int:
        """Number of elements in the complete flattened parameter vector."""

        return self._total_numel

    @property
    def local_shard_range(self) -> tuple[int, int]:
        """Flat range owned by this rank; the full range for world_size=1."""

        return 0, self._total_numel

    @property
    def local_state_numel(self) -> int:
        """Persistent elements used by the two local Adam state tensors."""

        return 2 * self._total_numel

    def _flat_state(self) -> dict[str, Any]:
        state = self.state[self._parameters[0]]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros(
                self._total_numel,
                device=self._device,
                dtype=self._dtype,
            )
            state["exp_avg_sq"] = torch.zeros(
                self._total_numel,
                device=self._device,
                dtype=self._dtype,
            )
        return state

    def _flatten_parameters(self) -> Tensor:
        return torch.cat([param.detach().reshape(-1) for param in self._parameters])

    def _flatten_gradients(self) -> Tensor:
        flat_gradients: list[Tensor] = []
        for index, param in enumerate(self._parameters):
            grad = param.grad
            if grad is None:
                raise RuntimeError(
                    "ZeroAdam requires every trainable parameter to receive a gradient; "
                    f"parameter {index} with shape {tuple(param.shape)} has grad=None."
                )
            if grad.is_sparse:
                raise RuntimeError("ZeroAdam does not support sparse gradients.")
            if grad.device != self._device or grad.dtype != self._dtype:
                raise RuntimeError(
                    "Gradient device and dtype must match the flattened parameters; "
                    f"parameter {index} has gradient {grad.device}/{grad.dtype}, "
                    f"expected {self._device}/{self._dtype}."
                )
            flat_gradients.append(grad.detach().reshape(-1))
        return torch.cat(flat_gradients)

    def _write_back_parameters(self, flat_parameters: Tensor) -> None:
        for param, flat_slice in zip(self._parameters, self._flat_slices):
            param.copy_(
                flat_parameters[flat_slice.start : flat_slice.end].view(flat_slice.shape)
            )

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Perform one Adam update on the complete local flat shard."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        flat_parameters = self._flatten_parameters()
        flat_gradients = self._flatten_gradients()
        if weight_decay != 0.0:
            flat_gradients.add_(flat_parameters, alpha=weight_decay)

        state = self._flat_state()
        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg.lerp_(flat_gradients, 1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            flat_gradients,
            flat_gradients,
            value=1.0 - beta2,
        )

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr / bias_correction1
        denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

        flat_parameters.addcdiv_(exp_avg, denominator, value=-step_size)
        self._write_back_parameters(flat_parameters)
        return loss
