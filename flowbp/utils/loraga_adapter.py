"""LoRA-GA initialization built on GradLoRA's dense-gradient collection.

LoRA-GA uses a rank-``2r`` randomized SVD of each globally averaged virtual
dense gradient.  The first ``r`` right singular vectors initialize A and the
next ``r`` left singular vectors initialize B.  The base weight is compensated
for the non-zero adapter so finalization does not change the model function.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from flowbp.utils.gradlora_adapter import (
    GradLoRACalibrator,
    _active_adapter_name,
    _is_lora_linear,
    _model_context,
    write_json_report,
)
from flowbp.utils.logging_ import main_print
from flowbp.utils.peft_adapter import (
    _split_target_modules,
    _validate_target_modules,
)


LORAGA_INIT_A_ATTR = "_loraga_init_A"
LORAGA_INIT_B_ATTR = "_loraga_init_B"
LORAGA_GAMMA_ATTR = "_loraga_stable_gamma"


@dataclass
class LoRAGALayerDiagnostic:
    name: str
    rows: int
    cols: int
    rank: int
    calibration_calls: int
    initialization: str
    gradient_fro_norm: float
    captured_energy: float
    sigma_max: float
    sigma_min_kept: float
    gamma: float
    initializer_scale: float
    use_rslora: bool
    # Energy of the held-in gradient retained by A's rank-``r`` subspace alone.
    # ``captured_energy`` covers the rank-``2r`` mean spectrum instead, so only
    # this field is comparable between the mean and second-moment arms.
    a_captured_energy: float = 0.0
    second_moment_beta: float = 0.0


def convert_loraga_to_portable_factors(
    a_final: torch.Tensor,
    b_final: torch.Tensor,
    a_init: torch.Tensor,
    b_init: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rank-``2r`` factors for the update relative to the original base.

    LoRA-GA trains against ``W_base - scaling * B_init @ A_init``.  Concatenating
    the factors as below gives

    ``B_portable @ A_portable = B_final @ A_final - B_init @ A_init``.

    The caller must keep the original adapter scaling unchanged.
    """
    tensors = {
        "a_final": a_final,
        "b_final": b_final,
        "a_init": a_init,
        "b_init": b_init,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"{name} must be a rank-2 tensor")

    rank, in_features = a_final.shape
    out_features, b_rank = b_final.shape
    if b_rank != rank:
        raise ValueError(
            "Final LoRA factor ranks do not match: "
            f"A={tuple(a_final.shape)}, B={tuple(b_final.shape)}"
        )
    if tuple(a_init.shape) != (rank, in_features):
        raise ValueError(
            "Initial A shape does not match final A: "
            f"{tuple(a_init.shape)} != {tuple(a_final.shape)}"
        )
    if tuple(b_init.shape) != (out_features, rank):
        raise ValueError(
            "Initial B shape does not match final B: "
            f"{tuple(b_init.shape)} != {tuple(b_final.shape)}"
        )

    init_a = a_init.to(device=a_final.device, dtype=a_final.dtype)
    init_b = b_init.to(device=b_final.device, dtype=b_final.dtype)
    portable_a = torch.cat((a_final, init_a), dim=0)
    portable_b = torch.cat((b_final, -init_b), dim=1)
    return portable_a, portable_b


# A concise alias for checkpoint conversion call sites.
loraga_to_portable_factors = convert_loraga_to_portable_factors


def _adapter_scaling(module: nn.Module, adapter: str) -> float:
    scaling = getattr(module, "scaling", None)
    if isinstance(scaling, dict):
        scaling = scaling.get(adapter)
    if scaling is None:
        scaling = getattr(module, "_loraga_scaling", None)
    if isinstance(scaling, torch.Tensor):
        scaling = scaling.item()
    scaling = float(scaling)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError(
            f"LoRA-GA requires positive finite adapter scaling, got {scaling}"
        )
    return scaling


def _adapter_uses_rslora(module: nn.Module, adapter: str) -> bool:
    use_rslora = getattr(module, "use_rslora", None)
    if isinstance(use_rslora, dict):
        use_rslora = use_rslora.get(adapter, False)
    if use_rslora is None:
        use_rslora = getattr(module, "_loraga_use_rslora", False)
    return bool(use_rslora)


def _adapter_stable_gamma(module: nn.Module) -> float:
    gamma = float(getattr(module, LORAGA_GAMMA_ATTR, 16.0))
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(
            f"LoRA-GA stable_gamma must be positive and finite, got {gamma}"
        )
    return gamma


def apply_loraga_if_enabled(
    transformer: nn.Module,
    args,
) -> tuple[nn.Module, bool]:
    """Optionally wrap ``transformer`` with a LoRA-GA-ready PEFT adapter."""
    if not bool(getattr(args, "use_loraga", False)):
        return transformer, False

    from peft import LoraConfig, get_peft_model

    target_modules = _split_target_modules(
        getattr(args, "loraga_target_modules", None)
    )
    target_modules = _validate_target_modules(transformer, target_modules)
    rank = int(args.loraga_rank)
    alpha = float(args.loraga_alpha)
    dropout = float(getattr(args, "loraga_dropout", 0.0))
    use_rslora = bool(getattr(args, "loraga_use_rslora", False))
    stable_gamma = float(getattr(args, "loraga_stable_gamma", 16.0))
    if rank <= 0:
        raise ValueError(f"LoRA-GA rank must be positive, got {rank}")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"LoRA-GA alpha must be positive and finite, got {alpha}")
    if not math.isfinite(stable_gamma) or stable_gamma <= 0.0:
        raise ValueError(
            "LoRA-GA stable_gamma must be positive and finite, "
            f"got {stable_gamma}"
        )

    base_config = getattr(transformer, "config", None)
    main_print(
        "--> applying calibration-ready LoRA-GA adapter "
        f"(rank={rank}, alpha={alpha}, dropout={dropout}, "
        f"use_rslora={use_rslora}, stable_gamma={stable_gamma}, "
        f"targets={target_modules})"
    )
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type=None,
            use_rslora=use_rslora,
        ),
    )

    initialized = 0
    with torch.no_grad():
        for module in transformer.modules():
            if not _is_lora_linear(module):
                continue
            adapter = _active_adapter_name(module)
            module.lora_B[adapter].weight.zero_()
            module._loraga_scaling = (
                alpha / math.sqrt(rank) if use_rslora else alpha / rank
            )
            module._loraga_use_rslora = use_rslora
            setattr(module, LORAGA_GAMMA_ATTR, stable_gamma)
            initialized += 1

    if base_config is not None:
        try:
            transformer.config = base_config
        except Exception:
            pass
    transformer.print_trainable_parameters()
    main_print(f"--> LoRA-GA registered {initialized} target linear layers")
    return transformer, True


class LoRAGACalibrator(GradLoRACalibrator):
    """Collect GradLoRA virtual gradients and finalize LoRA-GA factors.

    With ``use_second_moment`` enabled, A is drawn from GradLoRA's shrinkage
    mixture of the mean-gradient subspace and the per-batch gradient energy
    instead of the plain mean-gradient SVD. B, the stable scaling, and the base
    weight compensation are unchanged, so the two settings isolate the effect
    of subspace selection under identical initialization dynamics.
    """

    def __init__(self, model: nn.Module, **kwargs):
        # Second moments are optional here, so do not require CUDA merely
        # because GradLoRACalibrator's generic default points there.
        kwargs.setdefault("moment_compute_device", "cpu")
        super().__init__(model, **kwargs)

    @torch.no_grad()
    def finalize(
        self,
        *,
        rank: int,
        oversample: int = 8,
        niter: int = 4,
        svd_device: str = "cuda",
        diagnostics_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LoRAGALayerDiagnostic]:
        """Initialize LoRA-GA from globally averaged dense gradients."""
        if self.use_second_moment and self.batch_gradients:
            raise RuntimeError(
                "LoRA-GA has an unfinished second-moment calibration batch; "
                "call finish_batch() after every calibration backward pass"
            )
        self.close()
        if rank <= 0:
            raise ValueError(f"LoRA-GA rank must be positive, got {rank}")
        if svd_device not in {"cpu", "cuda"}:
            raise ValueError(f"svd_device must be 'cpu' or 'cuda', got {svd_device}")
        if svd_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA SVD requested but CUDA is unavailable")
        if self.use_second_moment and rank > self.second_moment_rank:
            raise ValueError(
                "LoRA-GA rank exceeds the configured second-moment sketch rank "
                f"({rank} > {self.second_moment_rank})"
            )

        distributed = dist.is_available() and dist.is_initialized()
        process_rank = dist.get_rank() if distributed else 0
        communication_device = (
            torch.device("cuda", torch.cuda.current_device())
            if distributed and dist.get_backend() == "nccl"
            else torch.device(svd_device)
        )
        factor_device = torch.device(svd_device)

        initializers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        diagnostics: list[LoRAGALayerDiagnostic] = []
        fallback_layers: list[str] = []
        for name in sorted(self.modules):
            module = self.modules[name]
            rows = int(module.base_layer.out_features)
            cols = int(module.base_layer.in_features)
            if self.use_second_moment:
                (
                    gradient,
                    total_calls,
                    moment_sketch,
                    moment_trace,
                    probe,
                ) = self._pop_layer_state(name)
            else:
                # A structurally inactive layer never fires its gradient hook.
                # Treat it as a zero gradient and fall back to standard LoRA.
                local_gradient = self.gradients.pop(
                    name,
                    torch.zeros(
                        rows,
                        cols,
                        device=self.accumulation_device,
                        dtype=torch.float32,
                    ),
                )
                gradient, total_calls = self._distributed_mean(
                    local_gradient,
                    self.calls.get(name, 0),
                )
                moment_sketch = None
                probe = None
            if tuple(gradient.shape) != (rows, cols):
                raise RuntimeError(
                    f"LoRA-GA gradient shape mismatch for {name}: "
                    f"{tuple(gradient.shape)} != {(rows, cols)}"
                )

            adapter = _active_adapter_name(module)
            gamma = _adapter_stable_gamma(module)
            use_rslora = _adapter_uses_rslora(module, adapter)
            initializer_scale = rows**0.25 / math.sqrt(gamma)
            gradient_fro_sq = float(gradient.float().pow(2).sum().item())
            initialization = "gradient_svd"
            captured_energy = 0.0
            a_captured_energy = 0.0
            sigma_max = 0.0
            sigma_min_kept = 0.0
            init_a: torch.Tensor | None = None
            init_b: torch.Tensor | None = None

            if gradient_fro_sq <= 0.0:
                initialization = "standard_lora_fallback_zero_gradient"
                fallback_layers.append(name)
            elif 2 * rank > min(rows, cols):
                initialization = "standard_lora_fallback_insufficient_dimension"
                fallback_layers.append(name)
            elif process_rank == 0:
                matrix = gradient.to(factor_device, dtype=torch.float32)
                if not torch.isfinite(matrix).all():
                    raise FloatingPointError(
                        f"LoRA-GA gradient for {name} contains non-finite values"
                    )
                decomposition_rank = 2 * rank
                q = min(
                    decomposition_rank + max(int(oversample), 0),
                    rows,
                    cols,
                )
                left_vectors, singular_values, right_vectors = torch.svd_lowrank(
                    matrix,
                    q=q,
                    niter=int(niter),
                )
                order = singular_values.argsort(descending=True)
                singular_values = singular_values[order]
                left_vectors = left_vectors[:, order]
                right_vectors = right_vectors[:, order]
                a_basis = right_vectors[:, :rank]
                if self.use_second_moment:
                    eigensystem = None
                    if not self.sigma_is_degenerate(
                        moment_trace, gradient_fro_sq
                    ):
                        eigensystem = self._mixed_right_eigensystem(
                            matrix,
                            probe,
                            moment_sketch,
                            moment_trace,
                            gradient_fro_sq,
                            rank,
                        )
                    if eigensystem is None:
                        # A degenerate covariance or a rank-deficient Nystrom
                        # core both mean beta = 0 for this layer. Keep the
                        # mean-gradient subspace so every rank still receives
                        # one broadcast.
                        initialization = "gradient_svd_second_moment_fallback"
                    else:
                        a_basis = eigensystem[0]
                        initialization = "gradient_second_moment"
                init_a = (
                    initializer_scale * a_basis.transpose(0, 1).contiguous()
                )
                init_b = (
                    initializer_scale
                    * left_vectors[:, rank:decomposition_rank].contiguous()
                )
                a_captured_energy = float(
                    (matrix @ a_basis).pow(2).sum().item()
                    / max(gradient_fro_sq, 1e-24)
                )
                captured_energy = float(
                    singular_values[:decomposition_rank].pow(2).sum().item()
                    / max(gradient_fro_sq, 1e-24)
                )
                sigma_max = float(singular_values[0].item())
                sigma_min_kept = float(
                    singular_values[decomposition_rank - 1].item()
                )
            else:
                init_a = torch.empty(
                    rank,
                    cols,
                    device=communication_device,
                    dtype=torch.float32,
                )
                init_b = torch.empty(
                    rows,
                    rank,
                    device=communication_device,
                    dtype=torch.float32,
                )

            if distributed and init_a is not None and init_b is not None:
                init_a = init_a.to(communication_device)
                init_b = init_b.to(communication_device)
                dist.broadcast(init_a, src=0)
                dist.broadcast(init_b, src=0)
            if init_a is not None and init_b is not None:
                initializers[name] = (init_a.cpu(), init_b.cpu())

            if process_rank == 0:
                diagnostics.append(
                    LoRAGALayerDiagnostic(
                        name=name,
                        rows=rows,
                        cols=cols,
                        rank=rank,
                        calibration_calls=total_calls,
                        initialization=initialization,
                        gradient_fro_norm=math.sqrt(gradient_fro_sq),
                        captured_energy=captured_energy,
                        sigma_max=sigma_max,
                        sigma_min_kept=sigma_min_kept,
                        gamma=gamma,
                        initializer_scale=initializer_scale,
                        use_rslora=use_rslora,
                        a_captured_energy=a_captured_energy,
                        second_moment_beta=(
                            self.second_moment_beta
                            if self.use_second_moment
                            else 0.0
                        ),
                    )
                )
            del gradient

        with _model_context(self.model):
            for name, module in self.modules.items():
                adapter = _active_adapter_name(module)
                a_weight = module.lora_A[adapter].weight
                b_weight = module.lora_B[adapter].weight
                initializer = initializers.get(name)
                if initializer is None:
                    # Ordinary LoRA fallback: retain random A and restore zero B.
                    b_weight.zero_()
                else:
                    init_a, init_b = initializer
                    if tuple(a_weight.shape) != tuple(init_a.shape):
                        raise RuntimeError(
                            f"LoRA-GA A shape mismatch for {name}: "
                            f"{tuple(a_weight.shape)} != {tuple(init_a.shape)}"
                        )
                    if tuple(b_weight.shape) != tuple(init_b.shape):
                        raise RuntimeError(
                            f"LoRA-GA B shape mismatch for {name}: "
                            f"{tuple(b_weight.shape)} != {tuple(init_b.shape)}"
                        )
                    a_weight.copy_(init_a.to(a_weight.device, a_weight.dtype))
                    b_weight.copy_(init_b.to(b_weight.device, b_weight.dtype))
                    scaling = _adapter_scaling(module, adapter)
                    compensation = b_weight.float() @ a_weight.float()
                    module.base_layer.weight.sub_(
                        (scaling * compensation).to(
                            module.base_layer.weight.device,
                            module.base_layer.weight.dtype,
                        )
                    )

                # Plain tensor attributes deliberately remain on CPU when the
                # model moves and are available to checkpoint conversion code.
                setattr(
                    module,
                    LORAGA_INIT_A_ATTR,
                    a_weight.detach().float().cpu().clone(),
                )
                setattr(
                    module,
                    LORAGA_INIT_B_ATTR,
                    b_weight.detach().float().cpu().clone(),
                )

        if distributed:
            dist.barrier()
        if process_rank == 0 and diagnostics_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(diagnostics_path)),
                exist_ok=True,
            )
            payload: dict[str, Any] = {
                "method": "LoRA-GA",
                "rank": rank,
                "oversample": oversample,
                "power_iterations": niter,
                "use_second_moment": self.use_second_moment,
                "second_moment_beta": (
                    self.second_moment_beta if self.use_second_moment else 0.0
                ),
                "layers": [asdict(item) for item in diagnostics],
            }
            if metadata:
                payload.update(metadata)
            write_json_report(diagnostics_path, payload, indent=2)
            main_print(f"--> LoRA-GA diagnostics saved to {diagnostics_path}")

        if process_rank == 0 and fallback_layers:
            main_print(
                "[warning] LoRA-GA kept standard LoRA initialization for "
                f"{len(fallback_layers)} layer(s): "
                + ", ".join(fallback_layers)
            )
        median_a_energy = float("nan")
        aligned = [
            item
            for item in diagnostics
            if item.initialization.startswith("gradient_")
        ]
        if aligned:
            energies = sorted(item.a_captured_energy for item in aligned)
            median_a_energy = energies[len(energies) // 2]
        main_print(
            f"--> LoRA-GA initialized {len(initializers)}/{len(self.modules)} "
            f"layers (A subspace="
            f"{'second_moment' if self.use_second_moment else 'mean'}, "
            f"median A captured energy={median_a_energy:.4f})"
        )
        return diagnostics
