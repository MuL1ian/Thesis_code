"""Reward-gradient-aligned LoRA initialization for FlowBP.

GradLoRA wraps a transformer with an ordinary PEFT LoRA adapter, runs a short
calibration phase with optimizer updates disabled, and estimates the virtual
dense gradient of every target linear layer from its input and output gradient:

    grad_W = grad_output.T @ input

The top right singular vectors initialize LoRA A while LoRA B remains zero.
Consequently the model function is unchanged after calibration, but the first
trainable LoRA directions are aligned with the FlowBP reward gradient.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from flowbp.utils.logging_ import main_print
from flowbp.utils.peft_adapter import (
    _split_target_modules,
    _validate_target_modules,
)


@dataclass
class GradLoRALayerDiagnostic:
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
    a_row_norm_scale: float


def _active_adapter_name(module: nn.Module) -> str:
    active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        return active
    if isinstance(active, (list, tuple)) and active:
        return str(active[0])
    active = getattr(module, "active_adapters", None)
    if isinstance(active, (list, tuple)) and active:
        return str(active[0])
    return "default"


def _is_lora_linear(module: nn.Module) -> bool:
    return (
        hasattr(module, "base_layer")
        and isinstance(getattr(module, "base_layer"), nn.Linear)
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
    )


def _model_context(model: nn.Module):
    if isinstance(model, FSDP):
        return FSDP.summon_full_params(
            model,
            recurse=True,
            writeback=True,
            rank0_only=False,
        )
    return nullcontext()


def apply_gradlora_if_enabled(transformer: nn.Module, args):
    """Optionally wrap ``transformer`` with a calibration-ready PEFT LoRA."""
    if not bool(getattr(args, "use_gradlora", False)):
        return transformer, False

    from peft import LoraConfig, get_peft_model

    target_modules = _split_target_modules(
        getattr(args, "gradlora_target_modules", None)
    )
    target_modules = _validate_target_modules(transformer, target_modules)
    base_config = getattr(transformer, "config", None)
    rank = int(args.gradlora_rank)
    alpha = float(args.gradlora_alpha)
    dropout = float(args.gradlora_dropout)

    main_print(
        "--> applying calibration-ready GradLoRA adapter "
        f"(rank={rank}, alpha={alpha}, dropout={dropout}, "
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
        ),
    )

    # Preserve the conditioning of standard LoRA initialization. At finalize
    # time each orthonormal right singular vector is scaled to the original
    # random A factor's mean row norm. B stays exactly zero throughout
    # calibration, so the model function is unchanged.
    initialized = 0
    with torch.no_grad():
        for module in transformer.modules():
            if not _is_lora_linear(module):
                continue
            adapter = _active_adapter_name(module)
            a_weight = module.lora_A[adapter].weight
            b_weight = module.lora_B[adapter].weight
            row_norm = float(
                a_weight.detach().float().norm().item()
                / math.sqrt(max(a_weight.shape[0], 1))
            )
            module._gradlora_a_row_norm = row_norm
            b_weight.zero_()
            initialized += 1

    if base_config is not None:
        try:
            transformer.config = base_config
        except Exception:
            pass
    transformer.print_trainable_parameters()
    main_print(f"--> GradLoRA registered {initialized} target linear layers")
    return transformer, True


class GradLoRACalibrator:
    """Accumulate virtual dense gradients and initialize GradLoRA factors."""

    def __init__(
        self,
        model: nn.Module,
        accumulation_device: str = "cpu",
        offload_inputs: bool = False,
        use_second_moment: bool = False,
        second_moment_beta: float = 0.5,
        second_moment_rank: int = 32,
        second_moment_oversample: int = 8,
        moment_compute_device: str = "cuda",
    ):
        if accumulation_device not in {"cpu", "cuda"}:
            raise ValueError(
                "gradlora accumulation_device must be 'cpu' or 'cuda', "
                f"got {accumulation_device!r}"
            )
        if accumulation_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA accumulation requested but CUDA is unavailable")
        if moment_compute_device not in {"cpu", "cuda"}:
            raise ValueError(
                "gradlora moment_compute_device must be 'cpu' or 'cuda', "
                f"got {moment_compute_device!r}"
            )
        if (
            use_second_moment
            and moment_compute_device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA second-moment computation requested but CUDA is unavailable"
            )
        if use_second_moment and not 0.0 < float(second_moment_beta) <= 1.0:
            raise ValueError(
                "gradlora second_moment_beta must be in (0, 1] when enabled, "
                f"got {second_moment_beta}"
            )
        if second_moment_rank <= 0:
            raise ValueError(
                "gradlora second_moment_rank must be positive, "
                f"got {second_moment_rank}"
            )

        self.model = model
        self.offload_inputs = bool(offload_inputs)
        self.use_second_moment = bool(use_second_moment)
        self.second_moment_beta = float(second_moment_beta)
        self.second_moment_rank = int(second_moment_rank)
        self.second_moment_oversample = max(int(second_moment_oversample), 0)
        self.accumulation_device = torch.device(
            "cuda", torch.cuda.current_device()
        ) if accumulation_device == "cuda" else torch.device("cpu")
        self.moment_compute_device = (
            torch.device("cuda", torch.cuda.current_device())
            if moment_compute_device == "cuda"
            else torch.device("cpu")
        )
        self.gradients: dict[str, torch.Tensor] = {}
        self.batch_gradients: dict[str, torch.Tensor] = {}
        self.batch_counts: dict[str, int] = {}
        self.second_moment_sketches: dict[str, torch.Tensor] = {}
        self.second_moment_traces: dict[str, float] = {}
        self.second_moment_probes: dict[str, torch.Tensor] = {}
        self.calls: dict[str, int] = {}
        self.modules: dict[str, nn.Module] = {}
        self.handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if not _is_lora_linear(module):
                continue
            self.modules[name] = module
            self.calls[name] = 0
            self.batch_counts[name] = 0
            self.second_moment_traces[name] = 0.0

            def forward_hook(
                hooked_module,
                inputs,
                output,
                *,
                layer_name=name,
            ):
                del hooked_module
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    return
                tensor_output = output[0] if isinstance(output, tuple) else output
                if (
                    not isinstance(tensor_output, torch.Tensor)
                    or not tensor_output.requires_grad
                ):
                    return
                layer_input = inputs[0].detach()
                if self.offload_inputs:
                    layer_input = layer_input.to("cpu")

                def output_gradient_hook(grad_output):
                    self._accumulate(layer_name, layer_input, grad_output)
                    return grad_output

                tensor_output.register_hook(output_gradient_hook)

            self.handles.append(module.register_forward_hook(forward_hook))

        if not self.modules:
            raise ValueError(
                "GradLoRA calibration found no PEFT LoRA Linear layers. "
                "Apply the GradLoRA adapter before constructing the calibrator."
            )
        main_print(
            f"--> GradLoRA calibration hooks registered on "
            f"{len(self.modules)} layers (accumulate={self.accumulation_device}, "
            f"offload_inputs={self.offload_inputs}, "
            f"second_moment={self.use_second_moment})"
        )

    @torch.no_grad()
    def _accumulate(
        self,
        name: str,
        layer_input: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> None:
        x = layer_input.to(grad_output.device).reshape(
            -1, layer_input.shape[-1]
        ).float()
        dy = grad_output.detach().reshape(-1, grad_output.shape[-1]).float()
        if x.shape[0] != dy.shape[0]:
            raise RuntimeError(
                f"GradLoRA shape mismatch for {name}: "
                f"input rows={x.shape[0]}, grad rows={dy.shape[0]}"
            )
        dense_gradient = dy.transpose(0, 1) @ x
        dense_gradient = dense_gradient.to(self.accumulation_device)
        target = self.batch_gradients if self.use_second_moment else self.gradients
        if name not in target:
            target[name] = dense_gradient
        else:
            target[name].add_(dense_gradient)
        self.calls[name] += 1

    def _second_moment_probe(self, name: str, cols: int) -> torch.Tensor:
        probe = self.second_moment_probes.get(name)
        if probe is not None:
            return probe
        sketch_size = min(
            self.second_moment_rank + self.second_moment_oversample,
            cols,
        )
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        probe = torch.randn(
            cols,
            sketch_size,
            generator=generator,
            dtype=torch.float32,
        )
        probe.div_(math.sqrt(max(sketch_size, 1)))
        probe = probe.to(self.accumulation_device)
        self.second_moment_probes[name] = probe
        return probe

    @torch.no_grad()
    def finish_batch(self) -> None:
        """Commit one complete calibration batch and its gradient-energy sketch."""
        if not self.use_second_moment:
            return
        if not self.batch_gradients:
            # ReLU clip / inactive rewards can produce a zero graph with no
            # hooked layer gradients. Skip that batch instead of aborting.
            main_print(
                "[warning] GradLoRA second-moment batch captured no layer "
                "gradients; skipping this calibration batch"
            )
            return

        for name, batch_gradient in self.batch_gradients.items():
            if name not in self.gradients:
                self.gradients[name] = batch_gradient.clone()
            else:
                self.gradients[name].add_(batch_gradient)
            self.batch_counts[name] += 1

            compute_gradient = batch_gradient.to(
                self.moment_compute_device,
                dtype=torch.float32,
            )
            probe = self._second_moment_probe(
                name,
                batch_gradient.shape[1],
            ).to(self.moment_compute_device)
            sketch = compute_gradient.transpose(0, 1) @ (
                compute_gradient @ probe
            )
            sketch = sketch.to(self.accumulation_device)
            if name not in self.second_moment_sketches:
                self.second_moment_sketches[name] = sketch
            else:
                self.second_moment_sketches[name].add_(sketch)
            self.second_moment_traces[name] += float(
                compute_gradient.pow(2).sum().item()
            )

        self.batch_gradients.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _distributed_mean(
        self,
        gradient: torch.Tensor,
        call_count: int,
    ) -> tuple[torch.Tensor, int]:
        if not (dist.is_available() and dist.is_initialized()):
            return gradient / max(call_count, 1), call_count

        backend = dist.get_backend()
        if backend == "nccl":
            reduce_device = torch.device("cuda", torch.cuda.current_device())
        else:
            reduce_device = gradient.device
        reduced = gradient.to(reduce_device)
        count = torch.tensor(
            [call_count],
            dtype=torch.long,
            device=reduce_device,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total_calls = int(count.item())
        reduced.div_(max(total_calls, 1))
        return reduced, total_calls

    def _distributed_total_count(
        self,
        count: int,
        device: torch.device,
    ) -> int:
        if not (dist.is_available() and dist.is_initialized()):
            return count
        count_tensor = torch.tensor([count], dtype=torch.long, device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        return int(count_tensor.item())

    @staticmethod
    def _nystrom_eigh(
        probe: torch.Tensor,
        sketch: torch.Tensor,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Approximate a PSD eigensystem from the one-pass sketch ``C @ probe``."""
        core = probe.transpose(0, 1) @ sketch
        core = 0.5 * (core + core.transpose(0, 1))
        core_values, core_vectors = torch.linalg.eigh(core)
        max_core = float(core_values.max().item())
        tolerance = max(max_core * 1e-6, 1e-12)
        keep = core_values > tolerance
        if not bool(keep.any()):
            return None

        kept_values = core_values[keep]
        kept_vectors = core_vectors[:, keep]
        nystrom_factor = sketch @ (
            kept_vectors / kept_values.sqrt().unsqueeze(0)
        )
        vectors, singular_values, _ = torch.linalg.svd(
            nystrom_factor,
            full_matrices=False,
        )
        eigenvalues = singular_values.square()
        order = eigenvalues.argsort(descending=True)
        vectors = vectors[:, order]
        eigenvalues = eigenvalues[order]

        if vectors.shape[1] < rank:
            augmented = torch.cat((vectors, probe), dim=1)
            vectors = torch.linalg.qr(augmented, mode="reduced").Q
            padding = torch.zeros(
                vectors.shape[1] - eigenvalues.shape[0],
                device=eigenvalues.device,
                dtype=eigenvalues.dtype,
            )
            eigenvalues = torch.cat((eigenvalues, padding))
        return vectors[:, :rank], eigenvalues[:rank]

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
    ) -> list[GradLoRALayerDiagnostic]:
        """Average gradients, initialize A from top right singular vectors."""
        if self.use_second_moment and self.batch_gradients:
            raise RuntimeError(
                "GradLoRA has an unfinished second-moment calibration batch; "
                "call finish_batch() after every calibration backward pass"
            )
        self.close()
        if rank <= 0:
            raise ValueError(f"GradLoRA rank must be positive, got {rank}")
        if self.use_second_moment and rank > self.second_moment_rank:
            raise ValueError(
                "final GradLoRA rank exceeds the configured second-moment "
                f"sketch rank ({rank} > {self.second_moment_rank})"
            )
        if svd_device not in {"cpu", "cuda"}:
            raise ValueError(f"svd_device must be 'cpu' or 'cuda', got {svd_device}")
        if svd_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA SVD requested but CUDA is unavailable")

        distributed = dist.is_available() and dist.is_initialized()
        process_rank = dist.get_rank() if distributed else 0
        communication_device = (
            torch.device("cuda", torch.cuda.current_device())
            if distributed and dist.get_backend() == "nccl"
            else torch.device(svd_device)
        )
        factor_device = (
            torch.device("cuda", torch.cuda.current_device())
            if svd_device == "cuda"
            else torch.device("cpu")
        )

        initializers: dict[str, torch.Tensor] = {}
        diagnostics: list[GradLoRALayerDiagnostic] = []
        zero_gradient_layers: list[str] = []
        for name in sorted(self.modules):
            if name not in self.gradients or self.calls.get(name, 0) == 0:
                raise RuntimeError(
                    f"GradLoRA collected no gradient for {name}. "
                    "Check that calibration performed a differentiable backward pass."
                )
            if self.use_second_moment:
                local_batches = self.batch_counts.get(name, 0)
                if (
                    local_batches == 0
                    or name not in self.second_moment_sketches
                    or name not in self.second_moment_probes
                ):
                    raise RuntimeError(
                        f"GradLoRA collected no completed second-moment batch "
                        f"for {name}"
                    )
                gradient, total_batches = self._distributed_mean(
                    self.gradients.pop(name),
                    local_batches,
                )
                moment_sketch, sketch_batches = self._distributed_mean(
                    self.second_moment_sketches.pop(name),
                    local_batches,
                )
                moment_trace_tensor, trace_batches = self._distributed_mean(
                    torch.tensor(
                        self.second_moment_traces[name],
                        dtype=torch.float32,
                        device=self.accumulation_device,
                    ),
                    local_batches,
                )
                if not (
                    total_batches == sketch_batches == trace_batches
                ):
                    raise RuntimeError(
                        f"GradLoRA second-moment batch-count mismatch for {name}"
                    )
                count_device = (
                    torch.device("cuda", torch.cuda.current_device())
                    if distributed and dist.get_backend() == "nccl"
                    else gradient.device
                )
                total_calls = self._distributed_total_count(
                    self.calls[name],
                    count_device,
                )
                moment_trace = float(moment_trace_tensor.item())
                probe = self.second_moment_probes.pop(name)
            else:
                gradient, total_calls = self._distributed_mean(
                    self.gradients.pop(name),
                    self.calls[name],
                )
                moment_sketch = None
                moment_trace = 0.0
                probe = None
            rows, cols = gradient.shape
            layer_rank = min(rank, rows, cols)
            q = min(layer_rank + max(int(oversample), 0), rows, cols)
            gradient_fro_sq = float(gradient.float().pow(2).sum().item())
            objective_energy = gradient_fro_sq
            if self.use_second_moment:
                objective_energy = (
                    (1.0 - self.second_moment_beta) * gradient_fro_sq
                    + self.second_moment_beta * moment_trace
                )
                if (
                    not math.isfinite(objective_energy)
                    or not torch.isfinite(moment_sketch).all()
                ):
                    raise FloatingPointError(
                        f"GradLoRA second moment for {name} contains "
                        "non-finite values"
                    )
            row_scale = float(
                getattr(self.modules[name], "_gradlora_a_row_norm", 1.0)
            )
            init_a: torch.Tensor | None = None
            initialization = (
                "gradient_second_moment"
                if self.use_second_moment
                else "gradient_svd"
            )
            kept_energy = 0.0
            sigma_max = 0.0
            sigma_min_kept = 0.0

            if process_rank == 0:
                matrix = gradient.to(factor_device, dtype=torch.float32)
                if not torch.isfinite(matrix).all():
                    raise FloatingPointError(
                        f"GradLoRA gradient for {name} contains non-finite values"
                    )
                if objective_energy <= 0.0:
                    # Some SD3 modules are structurally inactive. In particular,
                    # the final context_pre_only block computes add_q_proj only
                    # for a context output that is discarded, so its exact
                    # gradient is always zero. Keep PEFT's original random A
                    # (and zero B) for such layers instead of aborting all ranks.
                    initialization = "standard_lora_fallback"
                    zero_gradient_layers.append(name)
                else:
                    if self.use_second_moment:
                        probe_matrix = probe.to(
                            factor_device,
                            dtype=torch.float32,
                        )
                        mean_sketch = matrix.transpose(0, 1) @ (
                            matrix @ probe_matrix
                        )
                        mixed_sketch = (
                            (1.0 - self.second_moment_beta) * mean_sketch
                            + self.second_moment_beta
                            * moment_sketch.to(factor_device, dtype=torch.float32)
                        )
                        eigensystem = self._nystrom_eigh(
                            probe_matrix,
                            mixed_sketch,
                            layer_rank,
                        )
                        if eigensystem is None:
                            # Rank-0-only numerical failure: fall back to the
                            # mean-gradient SVD so distributed ranks stay in sync.
                            initialization = "gradient_svd"
                            _, singular_values, right_vectors = torch.svd_lowrank(
                                matrix,
                                q=q,
                                niter=int(niter),
                            )
                            order = singular_values.argsort(descending=True)
                            singular_values = singular_values[order]
                            right_vectors = right_vectors[:, order]
                            init_a = (
                                right_vectors[:, :layer_rank]
                                .transpose(0, 1)
                                .contiguous()
                            )
                            kept_energy = float(
                                singular_values[:layer_rank].pow(2).sum().item()
                                / max(gradient_fro_sq, 1e-24)
                            )
                            sigma_max = float(singular_values[0].item())
                            sigma_min_kept = float(
                                singular_values[layer_rank - 1].item()
                            )
                        else:
                            right_vectors, eigenvalues = eigensystem
                            init_a = right_vectors.transpose(0, 1).contiguous()
                            kept_energy = float(
                                min(
                                    eigenvalues.sum().item()
                                    / max(objective_energy, 1e-24),
                                    1.0,
                                )
                            )
                            sigma_max = float(eigenvalues[0].sqrt().item())
                            sigma_min_kept = float(
                                eigenvalues[layer_rank - 1].sqrt().item()
                            )
                    else:
                        _, singular_values, right_vectors = torch.svd_lowrank(
                            matrix,
                            q=q,
                            niter=int(niter),
                        )
                        order = singular_values.argsort(descending=True)
                        singular_values = singular_values[order]
                        right_vectors = right_vectors[:, order]
                        init_a = (
                            right_vectors[:, :layer_rank]
                            .transpose(0, 1)
                            .contiguous()
                        )
                        kept_energy = float(
                            singular_values[:layer_rank].pow(2).sum().item()
                            / max(gradient_fro_sq, 1e-24)
                        )
                        sigma_max = float(singular_values[0].item())
                        sigma_min_kept = float(
                            singular_values[layer_rank - 1].item()
                        )
                    if init_a is not None:
                        init_a.mul_(row_scale)
            elif objective_energy > 0.0:
                init_a = torch.empty(
                    layer_rank,
                    cols,
                    device=communication_device,
                    dtype=torch.float32,
                )
            else:
                initialization = "standard_lora_fallback"

            # All ranks agree on objective_energy after the all-reduce above, so
            # either every rank has an init_a buffer or every rank falls back.
            if distributed and init_a is not None:
                init_a = init_a.to(communication_device)
                dist.broadcast(init_a, src=0)
            if init_a is not None:
                initializers[name] = init_a.cpu()

            if process_rank == 0:
                diagnostics.append(
                    GradLoRALayerDiagnostic(
                        name=name,
                        rows=rows,
                        cols=cols,
                        rank=layer_rank,
                        calibration_calls=total_calls,
                        initialization=initialization,
                        gradient_fro_norm=math.sqrt(gradient_fro_sq),
                        captured_energy=kept_energy,
                        sigma_max=sigma_max,
                        sigma_min_kept=sigma_min_kept,
                        a_row_norm_scale=row_scale,
                    )
                )
            del gradient

        with _model_context(self.model):
            for name, module in self.modules.items():
                adapter = _active_adapter_name(module)
                a_weight = module.lora_A[adapter].weight
                b_weight = module.lora_B[adapter].weight
                init_a = initializers.get(name)
                if init_a is None:
                    # Preserve standard PEFT A initialization for a structurally
                    # inactive layer. B remains zero, preserving model outputs.
                    b_weight.zero_()
                    continue
                if tuple(a_weight.shape) != tuple(init_a.shape):
                    raise RuntimeError(
                        f"GradLoRA A shape mismatch for {name}: "
                        f"parameter={tuple(a_weight.shape)}, init={tuple(init_a.shape)}"
                    )
                a_weight.copy_(init_a.to(a_weight.device, a_weight.dtype))
                b_weight.zero_()

        if distributed:
            dist.barrier()
        if process_rank == 0 and diagnostics_path:
            os.makedirs(os.path.dirname(os.path.abspath(diagnostics_path)), exist_ok=True)
            payload = {
                "method": "GradLoRA",
                "rank": rank,
                "oversample": oversample,
                "power_iterations": niter,
                "use_second_moment": self.use_second_moment,
                "second_moment_beta": (
                    self.second_moment_beta
                    if self.use_second_moment
                    else 0.0
                ),
                "layers": [asdict(item) for item in diagnostics],
            }
            if metadata:
                payload.update(metadata)
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            main_print(f"--> GradLoRA diagnostics saved to {diagnostics_path}")

        if process_rank == 0 and zero_gradient_layers:
            main_print(
                "[warning] GradLoRA kept standard LoRA initialization for "
                f"{len(zero_gradient_layers)} zero-gradient layer(s): "
                + ", ".join(zero_gradient_layers)
            )
        median_energy = float("nan")
        aligned_diagnostics = [
            item
            for item in diagnostics
            if item.initialization.startswith("gradient_")
        ]
        if aligned_diagnostics:
            energies = sorted(item.captured_energy for item in aligned_diagnostics)
            median_energy = energies[len(energies) // 2]
        main_print(
            f"--> GradLoRA gradient-aligned {len(initializers)}/{len(self.modules)} layers; "
            f"median captured gradient energy={median_energy:.4f}"
        )
        return diagnostics
