"""LoRA-Pro optimizer core.

This module implements Algorithm 2 from *LoRA-Pro: Are Low-Rank Adapters
Properly Optimized?* (ICLR 2025).  For ``W = W0 + scale * B @ A`` it:

1. corrects the ordinary LoRA gradients in FP32 and forms their equivalent
   dense gradient;
2. keeps Adam first/second moments on that dense gradient;
3. maps the dense Adam direction back to A/B and applies the paper's
   Sylvester gauge correction.

No ``m x m`` or ``n x n`` projection matrix is materialized.  In distributed
training every complete adapter gradient is reduced, while each dense Adam
state is owned by one rank and only the small A/B updates are broadcast.
Optimizer checkpoints are rank-local when dense-state sharding is enabled.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class LoRAPair:
    """One PEFT LoRA factor pair implementing ``scale * B @ A``."""

    name: str
    adapter: str
    module: nn.Module
    a: nn.Parameter
    b: nn.Parameter
    scale: float

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.b.ndim != 2:
            raise ValueError(
                f"{self.name}: LoRA A/B must be matrices, got "
                f"{tuple(self.a.shape)} and {tuple(self.b.shape)}"
            )
        if self.a.shape[0] != self.b.shape[1]:
            raise ValueError(
                f"{self.name}: LoRA rank mismatch: "
                f"A={tuple(self.a.shape)}, B={tuple(self.b.shape)}"
            )
        if not math.isfinite(self.scale) or self.scale == 0.0:
            raise ValueError(f"{self.name}: scale must be finite and nonzero")

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])

    @property
    def dense_shape(self) -> tuple[int, int]:
        return int(self.b.shape[0]), int(self.a.shape[1])


def _mapping_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value}
    if hasattr(value, "keys"):
        return {str(key) for key in value.keys()}
    return set()


def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return value[key]
    except (KeyError, TypeError):
        return default


def _factor_weight(value: Any, pair_name: str) -> nn.Parameter:
    parameter = value if isinstance(value, nn.Parameter) else getattr(value, "weight", None)
    if not isinstance(parameter, nn.Parameter):
        raise TypeError(f"{pair_name}: LoRA factor has no nn.Parameter weight")
    return parameter


def is_peft_lora_layer(module: nn.Module) -> bool:
    """Return whether ``module`` exposes PEFT-style ``lora_A``/``lora_B``."""

    a_keys = _mapping_keys(getattr(module, "lora_A", None))
    b_keys = _mapping_keys(getattr(module, "lora_B", None))
    return bool(a_keys & b_keys)


def discover_peft_lora_pairs(
    model: nn.Module,
    adapter_names: str | Iterable[str] | None = None,
    *,
    trainable_only: bool = True,
) -> list[LoRAPair]:
    """Discover PEFT LoRA A/B pairs without importing PEFT.

    PEFT's own per-adapter ``scaling`` value is used, so ordinary LoRA and
    rank-stabilized LoRA are both scale-correct.  DoRA is rejected because its
    magnitude branch is not represented by ``W0 + scale * B @ A``.
    """

    if isinstance(adapter_names, str):
        selected = {adapter_names}
    elif adapter_names is None:
        selected = None
    else:
        selected = {str(name) for name in adapter_names}

    pairs: list[LoRAPair] = []
    seen_parameters: set[int] = set()
    for module_name, module in model.named_modules():
        if not is_peft_lora_layer(module):
            continue
        lora_a = getattr(module, "lora_A")
        lora_b = getattr(module, "lora_B")
        adapters = _mapping_keys(lora_a) & _mapping_keys(lora_b)
        if selected is not None:
            adapters &= selected

        for adapter in sorted(adapters):
            display_name = f"{module_name or '<root>'}[{adapter}]"
            use_dora = _mapping_get(getattr(module, "use_dora", None), adapter, False)
            if bool(use_dora):
                raise ValueError(
                    f"{display_name}: DoRA is not compatible with LoRA-Pro's "
                    "W0 + scale * B @ A parameterization"
                )

            a = _factor_weight(_mapping_get(lora_a, adapter), display_name)
            b = _factor_weight(_mapping_get(lora_b, adapter), display_name)
            if trainable_only:
                if not (a.requires_grad or b.requires_grad):
                    continue
                if a.requires_grad != b.requires_grad:
                    raise ValueError(
                        f"{display_name}: paper-correct LoRA-Pro requires both "
                        "A and B to be trainable"
                    )
            if id(a) in seen_parameters or id(b) in seen_parameters:
                raise ValueError(f"{display_name}: a LoRA parameter appears in two pairs")
            seen_parameters.update((id(a), id(b)))

            scale_value = _mapping_get(getattr(module, "scaling", None), adapter)
            if scale_value is None:
                alpha = _mapping_get(getattr(module, "lora_alpha", None), adapter)
                if alpha is None:
                    raise ValueError(
                        f"{display_name}: PEFT layer has neither scaling nor lora_alpha"
                    )
                scale_value = float(alpha) / float(a.shape[0])
            if isinstance(scale_value, Tensor):
                if scale_value.numel() != 1:
                    raise ValueError(f"{display_name}: scaling must be scalar")
                scale_value = scale_value.detach().item()
            pairs.append(
                LoRAPair(
                    name=display_name,
                    adapter=adapter,
                    module=module,
                    a=a,
                    b=b,
                    scale=float(scale_value),
                )
            )

    pairs.sort(key=lambda pair: (pair.name, pair.adapter))
    if not pairs:
        requested = "" if selected is None else f" for adapters {sorted(selected)}"
        raise ValueError(f"No trainable PEFT LoRA A/B pairs found{requested}")
    return pairs


# A concise alias for callers that do not care about the PEFT implementation detail.
find_lora_pairs = discover_peft_lora_pairs


def _symmetrize(matrix: Tensor) -> Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got {tuple(matrix.shape)}")
    return 0.5 * (matrix + matrix.transpose(0, 1))


def _psd_scale(matrix: Tensor) -> Tensor:
    """Return a homogeneous scale for dimensionless damping."""

    size = matrix.shape[0]
    if size == 0:
        return matrix.new_tensor(1.0)
    spectral_bound = matrix.abs().sum(dim=1).max()
    mean_diagonal = matrix.diagonal().abs().mean()
    floor = matrix.new_tensor(torch.finfo(matrix.dtype).eps)
    return torch.maximum(torch.maximum(spectral_bound, mean_diagonal), floor)


def damped_psd_pinv(
    matrix: Tensor,
    *,
    damping: float = 1e-6,
    rcond: float | None = None,
) -> tuple[Tensor, Tensor, float]:
    """Return a stable FP32 pseudo-inverse and its regularized PSD matrix.

    ``damping`` is dimensionless: the diagonal shift is
    ``damping * scale(matrix)``.  At zero damping, unresolved eigenmodes are
    removed according to ``rcond`` instead of inverted.
    """

    if damping < 0.0 or not math.isfinite(damping):
        raise ValueError(f"damping must be finite and non-negative, got {damping}")
    source_device = matrix.device
    symmetric = _symmetrize(matrix.float())
    scale_tensor = _psd_scale(symmetric)
    shift = float(damping) * scale_tensor
    regularized = symmetric + torch.eye(
        symmetric.shape[0], device=source_device, dtype=torch.float32
    ) * shift
    eigenvalues, eigenvectors = torch.linalg.eigh(regularized)

    tolerance = (
        float(rcond)
        if rcond is not None
        else 10.0 * torch.finfo(torch.float32).eps * max(symmetric.shape[0], 1)
    )
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError(f"rcond must be finite and non-negative, got {rcond}")
    cutoff = scale_tensor * tolerance
    inverse_values = torch.where(
        eigenvalues > cutoff,
        eigenvalues.reciprocal(),
        torch.zeros_like(eigenvalues),
    )
    inverse = (eigenvectors * inverse_values.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    return _symmetrize(inverse), regularized, float(shift.item())


def solve_symmetric_psd_sylvester(
    left: Tensor,
    right: Tensor,
    rhs: Tensor,
    *,
    damping: float = 0.0,
    rcond: float | None = None,
) -> Tensor:
    """Solve ``left @ X + X @ right = rhs`` for symmetric PSD coefficients.

    The solve uses the two small Gram eigensystems.  Singular modes use the
    Moore-Penrose solution; optional damping is scale-aware on each side.
    """

    if damping < 0.0 or not math.isfinite(damping):
        raise ValueError(f"damping must be finite and non-negative, got {damping}")
    left_f = _symmetrize(left.float())
    right_f = _symmetrize(right.float())
    if rhs.ndim != 2 or rhs.shape != (left_f.shape[0], right_f.shape[0]):
        raise ValueError(
            "Sylvester shape mismatch: "
            f"left={tuple(left.shape)}, right={tuple(right.shape)}, "
            f"rhs={tuple(rhs.shape)}"
        )

    left_shift = float(damping) * _psd_scale(left_f)
    right_shift = float(damping) * _psd_scale(right_f)
    left_f = left_f + torch.eye(
        left_f.shape[0], device=left_f.device, dtype=left_f.dtype
    ) * left_shift
    right_f = right_f + torch.eye(
        right_f.shape[0], device=right_f.device, dtype=right_f.dtype
    ) * right_shift
    left_values, left_vectors = torch.linalg.eigh(left_f)
    right_values, right_vectors = torch.linalg.eigh(right_f)
    # Gram roundoff can create tiny negative eigenvalues.
    left_values = left_values.clamp_min(0.0)
    right_values = right_values.clamp_min(0.0)

    transformed_rhs = (
        left_vectors.transpose(0, 1) @ rhs.float() @ right_vectors
    )
    denominators = left_values.unsqueeze(1) + right_values.unsqueeze(0)
    tolerance = (
        float(rcond)
        if rcond is not None
        else 10.0
        * torch.finfo(torch.float32).eps
        * max(left_f.shape[0], right_f.shape[0], 1)
    )
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError(f"rcond must be finite and non-negative, got {rcond}")
    scale = torch.maximum(
        denominators.max(),
        denominators.new_tensor(torch.finfo(torch.float32).eps),
    )
    solution_eigenbasis = torch.where(
        denominators > tolerance * scale,
        transformed_rhs / denominators.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(transformed_rhs),
    )
    return left_vectors @ solution_eigenbasis @ right_vectors.transpose(0, 1)


def equivalent_dense_gradient(
    b: Tensor,
    a: Tensor,
    update_a: Tensor,
    update_b: Tensor,
    scale: float,
) -> Tensor:
    """Form ``scale * (B @ update_A + update_B @ A)`` in FP32."""

    return float(scale) * (
        b.float() @ update_a.float() + update_b.float() @ a.float()
    )


@dataclass
class _CorrectionGeometry:
    a: Tensor
    b: Tensor
    scale: float
    gram_a_inverse: Tensor
    gram_b_inverse: Tensor
    gram_a_regularized: Tensor
    gram_b_regularized: Tensor
    damping_a: float
    damping_b: float
    condition_a: float
    condition_b: float


def _correction_geometry(
    b: Tensor,
    a: Tensor,
    scale: float,
    *,
    damping: float,
    rcond: float | None,
) -> _CorrectionGeometry:
    a_f = a.detach().float()
    b_f = b.detach().float()
    gram_b = b_f.transpose(0, 1) @ b_f
    gram_a = a_f @ a_f.transpose(0, 1)
    inverse_b, regularized_b, damping_b = damped_psd_pinv(
        gram_b, damping=damping, rcond=rcond
    )
    inverse_a, regularized_a, damping_a = damped_psd_pinv(
        gram_a, damping=damping, rcond=rcond
    )
    return _CorrectionGeometry(
        a=a_f,
        b=b_f,
        scale=float(scale),
        gram_a_inverse=inverse_a,
        gram_b_inverse=inverse_b,
        gram_a_regularized=regularized_a,
        gram_b_regularized=regularized_b,
        damping_a=damping_a,
        damping_b=damping_b,
        condition_a=float(torch.linalg.cond(regularized_a).item()),
        condition_b=float(torch.linalg.cond(regularized_b).item()),
    )


def _paper_particular_solution(
    geometry: _CorrectionGeometry,
    raw_a: Tensor,
    raw_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Theorem 2.1 with X=0, without constructing ``I - B B+``."""

    raw_a_f = raw_a.float()
    raw_b_f = raw_b.float()
    inverse_scale_squared = 1.0 / (geometry.scale * geometry.scale)
    update_a = (
        geometry.gram_b_inverse @ raw_a_f
    ) * inverse_scale_squared
    projected_b = raw_b_f - geometry.b @ (
        geometry.gram_b_inverse @ (geometry.b.transpose(0, 1) @ raw_b_f)
    )
    update_b = (
        projected_b @ geometry.gram_a_inverse
    ) * inverse_scale_squared
    return update_a, update_b


def raw_lora_grads_to_dense(
    b: Tensor,
    a: Tensor,
    raw_grad_a: Tensor,
    raw_grad_b: Tensor,
    scale: float,
    *,
    damping: float = 1e-6,
    rcond: float | None = None,
) -> Tensor:
    """Apply the paper's first correction and form the dense gradient."""

    geometry = _correction_geometry(
        b, a, scale, damping=damping, rcond=rcond
    )
    update_a, update_b = _paper_particular_solution(
        geometry, raw_grad_a, raw_grad_b
    )
    return equivalent_dense_gradient(
        geometry.b, geometry.a, update_a, update_b, geometry.scale
    )


def dense_gradient_to_lora_updates(
    b: Tensor,
    a: Tensor,
    dense_gradient: Tensor,
    scale: float,
    *,
    damping: float = 1e-6,
    rcond: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Map a dense direction to paper-optimal A/B updates.

    This is Algorithm 2's second correction.  The Sylvester term chooses the
    gauge closest to ordinary LoRA gradients and cancels exactly in the
    equivalent dense update.
    """

    geometry = _correction_geometry(
        b, a, scale, damping=damping, rcond=rcond
    )
    dense_f = dense_gradient.float()
    raw_a = geometry.scale * (geometry.b.transpose(0, 1) @ dense_f)
    raw_b = geometry.scale * (dense_f @ geometry.a.transpose(0, 1))
    update_a, update_b = _paper_particular_solution(geometry, raw_a, raw_b)
    rhs = -(update_a @ geometry.a.transpose(0, 1))
    gauge = solve_symmetric_psd_sylvester(
        geometry.gram_b_regularized,
        geometry.gram_a_regularized,
        rhs,
        damping=0.0,
        rcond=rcond,
    )
    return update_a + gauge @ geometry.a, update_b - geometry.b @ gauge


class LoRAProOptimizer(Optimizer):
    """Adam LoRA-Pro optimizer with optional owner-rank dense-state sharding.

    Args:
        model_or_pairs: a PEFT model or pairs returned by
            :func:`discover_peft_lora_pairs`.
        all_reduce_grads: all-reduce complete A/B gradients before clipping.
            Reduction is an average by default and is harmless if DDP already
            produced identical gradients.
        shard_dense_states: assign pair ``i`` to rank ``i % world_size``.
            Checkpoints returned by :meth:`state_dict` are then rank-local and
            must be saved/restored once per rank with the same world size.
    """

    def __init__(
        self,
        model_or_pairs: nn.Module | Sequence[LoRAPair],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        damping: float = 1e-6,
        rcond: float | None = None,
        max_grad_norm: float | None = None,
        error_if_nonfinite: bool = True,
        adapter_names: str | Iterable[str] | None = None,
        process_group: Any = None,
        all_reduce_grads: bool = True,
        average_gradients: bool = True,
        shard_dense_states: bool = True,
    ) -> None:
        if lr < 0.0 or not math.isfinite(lr):
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0 or not math.isfinite(eps):
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay != 0.0:
            raise ValueError(
                "LoRAProOptimizer only accepts weight_decay=0. "
                "Paper-correct decay also modifies the frozen base weight."
            )
        if damping < 0.0 or not math.isfinite(damping):
            raise ValueError(f"Invalid damping: {damping}")
        if max_grad_norm is not None and (
            max_grad_norm <= 0.0 or not math.isfinite(max_grad_norm)
        ):
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")

        if isinstance(model_or_pairs, nn.Module):
            pairs = discover_peft_lora_pairs(
                model_or_pairs, adapter_names=adapter_names
            )
        else:
            pairs = list(model_or_pairs)
            if not pairs or not all(isinstance(pair, LoRAPair) for pair in pairs):
                raise TypeError("model_or_pairs must be a model or non-empty LoRAPair list")
            pairs.sort(key=lambda pair: (pair.name, pair.adapter))

        parameters: list[nn.Parameter] = []
        parameter_ids: set[int] = set()
        for pair in pairs:
            for parameter in (pair.a, pair.b):
                if id(parameter) in parameter_ids:
                    raise ValueError(f"{pair.name}: duplicate optimizer parameter")
                parameter_ids.add(id(parameter))
                parameters.append(parameter)

        defaults = dict(
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=0.0,
        )
        super().__init__(parameters, defaults)
        self.pairs = pairs
        self.damping = float(damping)
        self.rcond = rcond
        self.max_grad_norm = max_grad_norm
        self.error_if_nonfinite = bool(error_if_nonfinite)
        self.process_group = process_group
        self.all_reduce_grads = bool(all_reduce_grads)
        self.average_gradients = bool(average_gradients)
        self.shard_dense_states = bool(shard_dense_states)
        self._owner_by_index = self._balanced_owner_assignment()
        self.last_diagnostics: dict[str, Any] = {
            "step": 0,
            "active_pairs": 0,
            "raw_grad_norm": 0.0,
            "clip_coefficient": 1.0,
        }

    def _distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _rank_world(self) -> tuple[int, int]:
        if not self._distributed():
            return 0, 1
        return (
            dist.get_rank(group=self.process_group),
            dist.get_world_size(group=self.process_group),
        )

    def _owner(self, pair_index: int) -> int:
        return self._owner_by_index[pair_index]

    def _balanced_owner_assignment(self) -> list[int]:
        """Greedily balance dense Adam-state elements across owner ranks."""
        _, world_size = self._rank_world()
        if not self.shard_dense_states or world_size == 1:
            return [0] * len(self.pairs)
        loads = [0] * world_size
        assignment = [0] * len(self.pairs)
        weighted_pairs = sorted(
            enumerate(self.pairs),
            key=lambda item: (
                -(2 * math.prod(item[1].dense_shape)),
                item[1].name,
            ),
        )
        for index, pair in weighted_pairs:
            owner = min(
                range(world_size),
                key=lambda candidate: (loads[candidate], candidate),
            )
            assignment[index] = owner
            loads[owner] += 2 * math.prod(pair.dense_shape)
        return assignment

    def _global_owner(self, owner: int) -> int:
        if self.process_group is None:
            return owner
        get_global_rank = getattr(dist, "get_global_rank", None)
        if get_global_rank is None:
            raise RuntimeError(
                "Custom process groups require torch.distributed.get_global_rank"
            )
        return int(get_global_rank(self.process_group, owner))

    def _group_for_pair(self, pair: LoRAPair) -> dict[str, Any]:
        for group in self.param_groups:
            if any(parameter is pair.a for parameter in group["params"]):
                if float(group.get("weight_decay", 0.0)) != 0.0:
                    raise ValueError(
                        "LoRAProOptimizer only accepts weight_decay=0"
                    )
                return group
        raise RuntimeError(f"{pair.name}: optimizer parameter group was removed")

    def _reduced_gradient(self, parameter: nn.Parameter) -> tuple[Tensor, bool]:
        if parameter.grad is not None and parameter.grad.is_sparse:
            raise RuntimeError("LoRA-Pro does not support sparse gradients")
        local_present = parameter.grad is not None
        gradient = (
            parameter.grad.detach().float().clone()
            if local_present
            else torch.zeros_like(parameter, dtype=torch.float32)
        )
        if not (self._distributed() and self.all_reduce_grads):
            return gradient, local_present

        present = torch.tensor(
            int(local_present), device=parameter.device, dtype=torch.int32
        )
        dist.all_reduce(present, op=dist.ReduceOp.SUM, group=self.process_group)
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=self.process_group)
        _, world_size = self._rank_world()
        if self.average_gradients:
            gradient.div_(world_size)
        return gradient, bool(present.item())

    def _collect_raw_gradients(
        self,
    ) -> list[tuple[int, LoRAPair, Tensor, Tensor]]:
        collected: list[tuple[int, LoRAPair, Tensor, Tensor]] = []
        for index, pair in enumerate(self.pairs):
            grad_a, present_a = self._reduced_gradient(pair.a)
            grad_b, present_b = self._reduced_gradient(pair.b)
            if not (present_a or present_b):
                continue
            collected.append((index, pair, grad_a, grad_b))
        return collected

    @staticmethod
    def _raw_norm(
        gradients: Sequence[tuple[int, LoRAPair, Tensor, Tensor]]
    ) -> float:
        squared_norm = math.fsum(
            float(grad.double().square().sum().item())
            for _, _, grad_a, grad_b in gradients
            for grad in (grad_a, grad_b)
        )
        return math.sqrt(squared_norm)

    def _dense_adam_direction(
        self,
        pair: LoRAPair,
        dense_gradient: Tensor,
        group: dict[str, Any],
    ) -> tuple[Tensor, int]:
        state = self.state[pair.a]
        if not state:
            state["step"] = 0
            state["dense_exp_avg"] = torch.zeros_like(
                dense_gradient, dtype=torch.float32
            )
            state["dense_exp_avg_sq"] = torch.zeros_like(
                dense_gradient, dtype=torch.float32
            )
        state["step"] = int(state["step"]) + 1
        step = int(state["step"])
        first, second = group["betas"]
        exp_avg = state["dense_exp_avg"]
        exp_avg_sq = state["dense_exp_avg_sq"]
        if (
            exp_avg.shape != dense_gradient.shape
            or exp_avg_sq.shape != dense_gradient.shape
        ):
            raise RuntimeError(f"{pair.name}: incompatible dense Adam state shape")
        exp_avg.mul_(first).add_(dense_gradient, alpha=1.0 - first)
        exp_avg_sq.mul_(second).addcmul_(
            dense_gradient, dense_gradient, value=1.0 - second
        )
        bias_correction1 = 1.0 - first**step
        bias_correction2 = 1.0 - second**step
        denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2))
        denominator.add_(group["eps"])
        return exp_avg.div(bias_correction1) / denominator, step

    def _broadcast_updates(
        self,
        update_a: Tensor,
        update_b: Tensor,
        owner: int,
    ) -> None:
        if not (self._distributed() and self.shard_dense_states):
            return
        source = self._global_owner(owner)
        dist.broadcast(update_a, src=source, group=self.process_group)
        dist.broadcast(update_b, src=source, group=self.process_group)

    def _aggregate_diagnostics(self, values: Tensor) -> Tensor:
        if not self._distributed() or not self.shard_dense_states:
            return values
        # Norm squares add across owners; damping and step are maxima.
        dist.all_reduce(
            values[:3], op=dist.ReduceOp.SUM, group=self.process_group
        )
        dist.all_reduce(
            values[3:], op=dist.ReduceOp.MAX, group=self.process_group
        )
        return values

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Perform one LoRA-Pro Adam step and return the optional closure loss."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        gradients = self._collect_raw_gradients()
        raw_norm = self._raw_norm(gradients)
        if not math.isfinite(raw_norm) and self.error_if_nonfinite:
            raise FloatingPointError(
                f"LoRA-Pro raw adapter gradient norm is non-finite: {raw_norm}"
            )
        clip_coefficient = 1.0
        if self.max_grad_norm is not None and math.isfinite(raw_norm):
            clip_coefficient = min(
                1.0, float(self.max_grad_norm) / (raw_norm + 1e-12)
            )

        rank, _ = self._rank_world()
        equivalent_norm_sq = 0.0
        update_a_norm_sq = 0.0
        update_b_norm_sq = 0.0
        maximum_damping = 0.0
        maximum_condition = 0.0
        maximum_mapping_residual = 0.0
        maximum_pair_step = int(self.last_diagnostics.get("step", 0))

        for index, pair, raw_a, raw_b in gradients:
            raw_a.mul_(clip_coefficient)
            raw_b.mul_(clip_coefficient)
            owner = self._owner(index)
            compute_here = not self.shard_dense_states or rank == owner
            update_a = torch.empty_like(pair.a, dtype=torch.float32)
            update_b = torch.empty_like(pair.b, dtype=torch.float32)

            if compute_here:
                geometry = _correction_geometry(
                    pair.b,
                    pair.a,
                    pair.scale,
                    damping=self.damping,
                    rcond=self.rcond,
                )
                corrected_a, corrected_b = _paper_particular_solution(
                    geometry, raw_a, raw_b
                )
                equivalent = equivalent_dense_gradient(
                    geometry.b,
                    geometry.a,
                    corrected_a,
                    corrected_b,
                    geometry.scale,
                )
                group = self._group_for_pair(pair)
                dense_direction, pair_step = self._dense_adam_direction(
                    pair, equivalent, group
                )
                final_a, final_b = dense_gradient_to_lora_updates(
                    geometry.b,
                    geometry.a,
                    dense_direction,
                    geometry.scale,
                    damping=self.damping,
                    rcond=self.rcond,
                )
                update_a.copy_(final_a)
                update_b.copy_(final_b)
                mapped_direction = equivalent_dense_gradient(
                    geometry.b,
                    geometry.a,
                    final_a,
                    final_b,
                    geometry.scale,
                )
                mapping_residual = float(
                    (mapped_direction - dense_direction).norm().item()
                    / max(float(dense_direction.norm().item()), 1e-24)
                )
                equivalent_norm_sq += float(equivalent.square().sum().item())
                update_a_norm_sq += float(final_a.square().sum().item())
                update_b_norm_sq += float(final_b.square().sum().item())
                maximum_damping = max(
                    maximum_damping,
                    geometry.damping_a,
                    geometry.damping_b,
                )
                maximum_condition = max(
                    maximum_condition,
                    geometry.condition_a,
                    geometry.condition_b,
                )
                maximum_mapping_residual = max(
                    maximum_mapping_residual,
                    mapping_residual,
                )
                maximum_pair_step = max(maximum_pair_step, pair_step)

            self._broadcast_updates(update_a, update_b, owner)
            group = self._group_for_pair(pair)
            learning_rate = float(group["lr"])
            pair.a.add_(
                update_a.to(device=pair.a.device, dtype=pair.a.dtype),
                alpha=-learning_rate,
            )
            pair.b.add_(
                update_b.to(device=pair.b.device, dtype=pair.b.dtype),
                alpha=-learning_rate,
            )

        diagnostic_device = (
            gradients[0][1].a.device if gradients else self.pairs[0].a.device
        )
        diagnostic_values = torch.tensor(
            [
                equivalent_norm_sq,
                update_a_norm_sq,
                update_b_norm_sq,
                maximum_damping,
                maximum_condition,
                maximum_mapping_residual,
                float(maximum_pair_step),
            ],
            device=diagnostic_device,
            dtype=torch.float64,
        )
        diagnostic_values = self._aggregate_diagnostics(diagnostic_values)
        values = diagnostic_values.cpu().tolist()
        self.last_diagnostics = {
            "step": int(values[6]),
            "active_pairs": len(gradients),
            "owned_active_pairs": sum(
                1
                for index, _, _, _ in gradients
                if not self.shard_dense_states or self._owner(index) == rank
            ),
            "raw_grad_norm": raw_norm,
            "clip_coefficient": clip_coefficient,
            "equivalent_grad_norm": math.sqrt(max(values[0], 0.0)),
            "update_a_norm": math.sqrt(max(values[1], 0.0)),
            "update_b_norm": math.sqrt(max(values[2], 0.0)),
            "maximum_absolute_damping": values[3],
            "maximum_gram_condition": values[4],
            "maximum_mapping_residual": values[5],
            "dense_state_elements_local": sum(
                state_value.numel()
                for state in self.state.values()
                for key, state_value in state.items()
                if key in {"dense_exp_avg", "dense_exp_avg_sq"}
            ),
        }
        return loss

    def get_step_diagnostics(self) -> dict[str, Any]:
        """Return a detached copy of the most recent step diagnostics."""

        return copy.deepcopy(self.last_diagnostics)

    def state_dict(self) -> dict[str, Any]:
        """Return this rank's optimizer shard plus LoRA-Pro metadata."""

        result = super().state_dict()
        rank, world_size = self._rank_world()
        result["lorapro_metadata"] = {
            "version": 1,
            "rank": rank,
            "world_size": world_size,
            "shard_dense_states": self.shard_dense_states,
            "pair_names": [pair.name for pair in self.pairs],
            "owner_by_index": list(self._owner_by_index),
            "diagnostics": copy.deepcopy(self.last_diagnostics),
        }
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a rank-local optimizer shard created by :meth:`state_dict`."""

        # Avoid duplicating potentially large dense moments before the base
        # optimizer performs its own device/dtype restoration.
        state_copy = dict(state_dict)
        metadata = state_copy.pop("lorapro_metadata", None)
        if metadata is not None:
            rank, world_size = self._rank_world()
            expected_names = [pair.name for pair in self.pairs]
            if metadata.get("pair_names") != expected_names:
                raise ValueError("LoRA-Pro checkpoint pair names/order do not match")
            if metadata.get("owner_by_index") != self._owner_by_index:
                raise ValueError("LoRA-Pro checkpoint owner assignment does not match")
            if bool(metadata.get("shard_dense_states")) != self.shard_dense_states:
                raise ValueError("LoRA-Pro checkpoint sharding mode does not match")
            if self.shard_dense_states and (
                int(metadata.get("rank", -1)) != rank
                or int(metadata.get("world_size", -1)) != world_size
            ):
                raise ValueError(
                    "Sharded LoRA-Pro state must be restored on the same rank "
                    "and world size"
                )
        super().load_state_dict(state_copy)
        # torch.optim casts floating states to a BF16/FP16 parameter dtype.
        # Restore moments directly from the checkpoint so precision is not
        # lost in a BF16 -> FP32 round trip.
        saved_parameter_ids = [
            parameter_id
            for group in state_copy["param_groups"]
            for parameter_id in group["params"]
        ]
        current_parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        ]
        parameter_map = dict(zip(saved_parameter_ids, current_parameters))
        for saved_id, saved_state in state_copy["state"].items():
            parameter = parameter_map.get(saved_id)
            if parameter is None:
                continue
            current_state = self.state[parameter]
            for key in ("dense_exp_avg", "dense_exp_avg_sq"):
                source = saved_state.get(key)
                if isinstance(source, Tensor):
                    current_state[key] = source.detach().to(
                        device=parameter.device,
                        dtype=torch.float32,
                        copy=True,
                    )
        for state in self.state.values():
            for key in ("dense_exp_avg", "dense_exp_avg_sq"):
                if key in state:
                    state[key] = state[key].float()
        if metadata is not None and isinstance(metadata.get("diagnostics"), dict):
            self.last_diagnostics = copy.deepcopy(metadata["diagnostics"])


__all__ = [
    "LoRAPair",
    "LoRAProOptimizer",
    "damped_psd_pinv",
    "dense_gradient_to_lora_updates",
    "discover_peft_lora_pairs",
    "equivalent_dense_gradient",
    "find_lora_pairs",
    "is_peft_lora_layer",
    "raw_lora_grads_to_dense",
    "solve_symmetric_psd_sylvester",
]
