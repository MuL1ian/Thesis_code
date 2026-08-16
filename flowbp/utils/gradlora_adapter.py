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


GRADLORA_A_SCALE_MODES = ("legacy", "relative", "gain")


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``None`` so the payload is valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json_report(path: str, payload: Any, **dump_kwargs: Any) -> None:
    """Write a diagnostics report as strictly valid JSON.

    ``json.dump`` emits bare ``NaN`` by default, which is not JSON and is
    rejected by strict parsers -- including the analysis tools these reports
    exist to feed. Undefined diagnostics are genuinely null, so they are
    written that way and ``allow_nan=False`` keeps any survivor loud.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, allow_nan=False, **dump_kwargs)


@torch.no_grad()
def trainable_parameter_digest(model: nn.Module) -> str:
    """SHA-256 over the raw bytes of every trainable parameter on this rank.

    Bytes rather than norms: the probe is only sound if it leaves the model
    exactly as calibration left it, and a summary statistic can miss updates
    that cancel. Under ``use_orig_params=True`` each rank digests its own
    shard, which is sufficient because any real update would touch some shard.
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        # reshape(-1) first: a dtype view needs a contiguous last dimension,
        # and this also covers scalar parameters, which cannot be viewed.
        tensor = parameter.detach().to("cpu").contiguous().reshape(-1)
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def assert_lora_b_is_zero(model: nn.Module, *, context: str) -> None:
    """Fail loudly if any LoRA B is non-zero.

    B == 0 is what makes the adapter an exact identity, so the held-out probe
    measures the base model's gradients rather than gradients already bent by a
    partially trained adapter. A non-zero B silently invalidates every held-out
    number rather than producing an obvious error, so it is checked instead of
    assumed.
    """
    offenders = [
        name
        for name, parameter in model.named_parameters()
        if ".lora_B." in name and bool(parameter.detach().any().item())
    ]
    if offenders:
        raise RuntimeError(
            f"GradLoRA {context}: LoRA B is non-zero on "
            f"{len(offenders)} parameters (e.g. {offenders[0]}); the adapter "
            "is no longer an identity and held-out scores are invalid"
        )

# tr(Sigma) = (snr - 1) * ||Gbar||_F^2, so it collapses to zero exactly when the
# per-batch gradients agree. Below this fraction of ||Gbar||_F^2 the covariance
# term is numerical noise and the criterion falls back to beta = 0.
SIGMA_TRACE_TOL = 1e-6


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
    a_scale_mode: str = "legacy"
    write_gain: float = 0.0
    # tr(M) / ||Gbar||_F^2, at least 1 by Cauchy-Schwarz and read against
    # K_cal rather than bounded by it: 1 means the per-batch gradients agree,
    # about K_cal means independent zero-mean noise, and above K_cal means they
    # partly cancel, which is worse than noise. It says how far apart the two
    # endpoints of the mixed criterion can be and predicts the calibration
    # budget at which the selection stabilises.
    gradient_snr: float = float("nan")
    # sum(sigma_i) / (sqrt(r) * ||sigma||_2) in [r^-0.5, 1]. One means a flat
    # retained spectrum (worst case for the Nystrom sketch), r^-0.5 means a
    # single direction carries everything.
    spectral_flatness: float = float("nan")
    # Cosine between Adam's first B update and the ideal low-rank direction,
    # computed from the actual initializer rather than predicted from a
    # spectrum. B starts at zero, so the first B gradient is exactly Gbar A^T
    # and Adam's first step is its elementwise sign; this is therefore
    # ||M||_1 / (sqrt(numel) * ||M||_2) with M = Gbar A^T. It sits near
    # sqrt(2/pi) ~ 0.798 for a Gaussian-shaped M, which is the value a measured
    # first-update cosine has to be read against -- not one.
    predicted_adam_cosine: float = float("nan")


def _adapter_scaling(module: nn.Module, adapter: str) -> float:
    """Return the PEFT output scaling ``s`` applied to ``B @ A``."""
    scaling = getattr(module, "scaling", None)
    if isinstance(scaling, dict):
        scaling = scaling.get(adapter)
    if isinstance(scaling, torch.Tensor):
        scaling = scaling.item()
    if scaling is None:
        raise ValueError(
            f"GradLoRA could not read adapter scaling for {adapter!r}"
        )
    scaling = float(scaling)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError(
            f"GradLoRA requires positive finite adapter scaling, got {scaling}"
        )
    return scaling


def _resolve_a_row_scale(
    module: nn.Module,
    adapter: str,
    mode: str,
    scale: float,
) -> tuple[float, float]:
    """Return the ``(row_scale, write_gain)`` of this layer's A initializer.

    B is exactly zero at initialization, so ``W0 + s * B @ A`` equals ``W0``
    for every row scale and the scale costs nothing in function space. What it
    does set is the first-step write gain ``g = s * c``: Adam moves B by about
    the learning rate per coordinate irrespective of ``c``, so the induced
    weight update grows linearly in ``c``. Conversely A's own relative update
    rate falls as ``1 / c``, which drives large-``c`` runs toward the frozen-A
    regime, so useful values of ``g`` are bounded on both sides.
    """
    if mode not in GRADLORA_A_SCALE_MODES:
        raise ValueError(
            "gradlora a_scale_mode must be one of "
            f"{GRADLORA_A_SCALE_MODES}, got {mode!r}"
        )
    scaling = _adapter_scaling(module, adapter)
    legacy = float(getattr(module, "_gradlora_a_row_norm", 1.0))
    if mode == "legacy":
        row_scale = legacy
    else:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "gradlora a_scale must be finite and positive in "
                f"{mode!r} mode, got {scale}"
            )
        row_scale = scale * legacy if mode == "relative" else scale / scaling
    if not math.isfinite(row_scale) or row_scale <= 0.0:
        raise ValueError(
            "GradLoRA resolved a non-positive A row scale "
            f"({row_scale}) in {mode!r} mode"
        )
    return row_scale, scaling * row_scale


def _first_step_adam_cosine(
    mean_gradient: torch.Tensor,
    init_a: torch.Tensor | None,
) -> float:
    """Cosine between Adam's first ``B`` update and the ideal direction.

    ``B`` starts at zero, so the first gradient w.r.t. ``B`` is exactly
    ``M = Gbar A^T`` and Adam's first step is its elementwise sign (the
    per-parameter second moment equals the square of the first). The cosine
    between ``sign(M)`` and ``M`` is then ``||M||_1 / (sqrt(numel) ||M||_2)``.

    Measured from the initializer that will actually be written, not inferred
    from the criterion's spectrum: under second-moment mixing the eigenvectors
    of ``C_beta`` do not diagonalise ``Gbar^T Gbar``, so that spectrum carries
    no information about the shape of ``M``. Cheap because ``M`` is only
    ``d_out x r``.
    """
    if init_a is None:
        return float("nan")
    update = mean_gradient.to(init_a.device, dtype=torch.float32) @ (
        init_a.to(dtype=torch.float32).transpose(0, 1)
    )
    l2 = float(update.pow(2).sum().sqrt().item())
    if not math.isfinite(l2) or l2 <= 0.0:
        return float("nan")
    l1 = float(update.abs().sum().item())
    return l1 / (math.sqrt(update.numel()) * l2)


def _spectral_flatness(spectrum: torch.Tensor | None) -> float:
    """``sum(sigma_i) / (sqrt(r) * ||sigma||_2)`` over the retained spectrum.

    Ranges over ``[r**-0.5, 1]``: one when the retained directions carry equal
    weight, ``r**-0.5`` when a single direction carries everything.
    """
    if spectrum is None or spectrum.numel() == 0:
        return float("nan")
    values = spectrum.detach().float().abs()
    l2 = float(values.pow(2).sum().sqrt().item())
    if not math.isfinite(l2) or l2 <= 0.0:
        return float("nan")
    return float(values.sum().item()) / (math.sqrt(values.numel()) * l2)


def _apply_spectral_exponent(
    initializer: torch.Tensor,
    spectrum: torch.Tensor,
    exponent: float,
    target_row_rms: float,
) -> torch.Tensor:
    """Shape initializer rows by ``spectrum**exponent`` without changing scale.

    ``initializer`` contains orthonormal right spectral vectors as rows.
    The spectrum is normalized by its maximum before exponentiation, so the
    operation depends only on relative spectral decay. A final global row-RMS
    normalization preserves the target scale selected by the caller.
    """

    if not math.isfinite(exponent):
        raise ValueError(
            f"gradlora spectral_exponent must be finite, got {exponent}"
        )
    if target_row_rms <= 0.0 or not math.isfinite(target_row_rms):
        raise ValueError(
            "GradLoRA target A row RMS must be finite and positive, "
            f"got {target_row_rms}"
        )
    if initializer.ndim != 2:
        raise ValueError(
            f"GradLoRA A initializer must be a matrix, got {initializer.shape}"
        )

    result = initializer.float()
    values = spectrum[: result.shape[0]].to(result.device, dtype=torch.float32)
    if values.numel() != result.shape[0]:
        raise ValueError(
            "GradLoRA initializer/spectrum rank mismatch: "
            f"{result.shape[0]} != {values.numel()}"
        )
    if not torch.isfinite(values).all() or bool((values < 0).any()):
        raise FloatingPointError(
            "GradLoRA initializer spectrum must be finite and non-negative"
        )

    if exponent != 0.0:
        max_value = values.max()
        if float(max_value.item()) <= 0.0:
            raise ValueError(
                "Cannot apply a non-zero GradLoRA spectral exponent to a "
                "zero spectrum"
            )
        relative = values / max_value
        # Nyström may pad a rank-deficient eigensystem to preserve the requested
        # LoRA rank. Shape only resolved modes; padded rows retain standard
        # LoRA strength instead of becoming permanently dead A/B rank slots.
        valid = relative > 1e-6
        weights = torch.ones_like(relative)
        weights[valid] = relative[valid].pow(exponent)
        result = result * weights.unsqueeze(1)

    # Preserve the requested global row-RMS so p changes relative spectral
    # weighting without changing the adapter's overall initialization scale.
    current_row_rms = result.norm() / math.sqrt(max(result.shape[0], 1))
    if not torch.isfinite(current_row_rms) or float(current_row_rms.item()) <= 0.0:
        raise FloatingPointError(
            "GradLoRA spectral shaping produced a zero or non-finite A initializer"
        )
    return result * (float(target_row_rms) / current_row_rms)


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
        collect_probe_state: bool = False,
        a_scale_mode: str = "legacy",
        a_scale: float = 1.0,
    ):
        if a_scale_mode not in GRADLORA_A_SCALE_MODES:
            raise ValueError(
                "gradlora a_scale_mode must be one of "
                f"{GRADLORA_A_SCALE_MODES}, got {a_scale_mode!r}"
            )
        if a_scale_mode != "legacy" and (
            not math.isfinite(float(a_scale)) or float(a_scale) <= 0.0
        ):
            raise ValueError(
                "gradlora a_scale must be finite and positive in "
                f"{a_scale_mode!r} mode, got {a_scale}"
            )
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
        self.a_scale_mode = str(a_scale_mode)
        self.a_scale = float(a_scale)
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
        self.collect_probe_state = bool(collect_probe_state)
        self.probe_mode = False
        # variant -> layer name -> tensor / scalar
        self.probe_subspaces: dict[str, dict[str, torch.Tensor]] = {}
        self.probe_references: dict[str, dict[str, torch.Tensor]] = {}
        self.probe_reference_energy: dict[str, dict[str, float]] = {}
        self.probe_layer_stats: dict[str, dict[str, dict[str, float]]] = {}
        self.probe_batch_records: dict[str, list[dict[str, float]]] = {}
        # Mean calibration gradient per layer, kept only for probing. RQ3 needs
        # it to give the oracle arm a reference direction on the same footing
        # as every other arm. A full dense copy of every target layer, so it is
        # opt-in and normally requested on the scoring rank alone.
        self.keep_calibration_gradients = False
        self.probe_calibration_gradients: dict[str, torch.Tensor] = {}
        # Reducing per-layer gradients across ranks is a collective, so every
        # rank must agree on whether it happens. The observer that consumes the
        # result, in contrast, is normally attached on one rank only: it
        # retains every held-out batch until the oracle is known, and holding
        # that on all ranks would multiply the memory by the world size.
        self.probe_reduce_gradients = False
        # Optional callback receiving (batch_index, data-parallel-mean layer
        # gradients) once per held-out batch. Scoring that needs the full
        # per-layer gradient, rather than the running scalars accumulated here,
        # attaches through this rather than by extending this class.
        self.probe_observer: Any = None
        self._probe_batch_index = 0
        self._register_hooks()

    def _register_hooks(self, phase: str = "calibration") -> None:
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
            f"--> GradLoRA {phase} hooks registered on "
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
        target = (
            self.batch_gradients
            if self.use_second_moment or self.probe_mode
            else self.gradients
        )
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
        if self.probe_mode:
            self._finish_probe_batch()
            return
        if not self.use_second_moment:
            return
        if not self.batch_gradients:
            # ReLU clip / inactive rewards can produce a zero graph with no
            # hooked layer gradients. Other ranks may still have some, and the
            # reduction below is collective, so this rank contributes zeros
            # rather than returning.
            main_print(
                "[warning] GradLoRA second-moment batch captured no layer "
                "gradients on this rank; contributing zeros to this "
                "calibration batch"
            )

        # Sigma describes the spread between *global* batches, so the batch
        # gradient must be reduced across ranks before it is squared. Squaring
        # each rank's shard instead would measure the spread between shards --
        # a batch W times smaller, hence W times noisier -- which inflates snr,
        # biases the beta mixture toward the covariance term, and redefines the
        # batch away from the unit the held-out probe scores in.
        batch_gradients = self._reduced_batch_gradients()

        for name in sorted(self.modules):
            # Counted even when absent, because a global batch that produced no
            # gradient for this layer is a genuine zero observation of it, not
            # a missing one.
            self.batch_counts[name] += 1
            batch_gradient = batch_gradients.get(name)
            if batch_gradient is None:
                continue
            if name not in self.gradients:
                self.gradients[name] = batch_gradient.clone()
            else:
                self.gradients[name].add_(batch_gradient)

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

    def _layer_shape(self, name: str) -> tuple[int, int]:
        base_layer = self.modules[name].base_layer
        return int(base_layer.out_features), int(base_layer.in_features)

    def _zero_gradient(self, name: str) -> torch.Tensor:
        """A zero stand-in so an inactive layer can still join the reductions."""
        return torch.zeros(
            self._layer_shape(name),
            dtype=torch.float32,
            device=self.accumulation_device,
        )

    @torch.no_grad()
    def _pop_layer_state(
        self,
        name: str,
    ) -> tuple[torch.Tensor, int, torch.Tensor | None, float, torch.Tensor | None]:
        """Consume one layer's calibration buffers as globally averaged state.

        Returns the mean dense gradient and its total hook-call count, plus --
        only when second-moment mixing is enabled -- the averaged gradient
        energy sketch, its trace, and the layer's random probe.

        A layer that produced nothing on *this* rank contributes zeros and
        still takes part in every reduction. Raising instead would abort one
        rank while the others block forever in the collectives below, and it
        would also pre-empt the ``standard_lora_fallback`` that a globally
        inactive layer is supposed to reach -- the caller decides that, after
        the reduction, from the global gradient.
        """
        local_gradient = self.gradients.pop(name, None)
        if local_gradient is None:
            local_gradient = self._zero_gradient(name)

        if not self.use_second_moment:
            gradient, total_calls = self._distributed_mean(
                local_gradient,
                self.calls.get(name, 0),
            )
            return gradient, total_calls, None, 0.0, None

        local_batches = self.batch_counts.get(name, 0)
        probe = self._second_moment_probe(name, self._layer_shape(name)[1])
        local_sketch = self.second_moment_sketches.pop(name, None)
        if local_sketch is None:
            local_sketch = torch.zeros(
                (probe.shape[0], probe.shape[1]),
                dtype=torch.float32,
                device=self.accumulation_device,
            )
        # These three buffers are already identical on every rank, having been
        # accumulated from reduced batch gradients. Reducing them again is
        # idempotent rather than wrong -- both numerator and denominator pick
        # up the same factor of the world size -- and it keeps one code path
        # for both mixing modes while cross-checking that the ranks agree on
        # how many batches were committed.
        gradient, total_batches = self._distributed_mean(
            local_gradient,
            local_batches,
        )
        moment_sketch, sketch_batches = self._distributed_mean(
            local_sketch,
            local_batches,
        )
        moment_trace_tensor, trace_batches = self._distributed_mean(
            torch.tensor(
                self.second_moment_traces.get(name, 0.0),
                dtype=torch.float32,
                device=self.accumulation_device,
            ),
            local_batches,
        )
        if not (total_batches == sketch_batches == trace_batches):
            raise RuntimeError(
                f"GradLoRA second-moment batch-count mismatch for {name}"
            )
        distributed = dist.is_available() and dist.is_initialized()
        count_device = (
            torch.device("cuda", torch.cuda.current_device())
            if distributed and dist.get_backend() == "nccl"
            else gradient.device
        )
        total_calls = self._distributed_total_count(
            self.calls.get(name, 0),
            count_device,
        )
        self.second_moment_probes.pop(name, None)
        return (
            gradient,
            total_calls,
            moment_sketch,
            float(moment_trace_tensor.item()),
            probe,
        )

    @staticmethod
    @torch.no_grad()
    def _mean_right_eigensystem(
        matrix: torch.Tensor,
        q: int,
        niter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Descending top-``q`` right singular system of the mean gradient.

        This is the ``beta = 0`` criterion: the eigenvectors of
        ``C_0 = GbarT_Gbar / ||Gbar||_F^2`` are the right singular vectors of
        ``Gbar``, and its eigenvalues are ``sigma_i^2 / ||Gbar||_F^2``.
        """
        _, singular_values, right_vectors = torch.svd_lowrank(
            matrix,
            q=q,
            niter=int(niter),
        )
        order = singular_values.argsort(descending=True)
        return right_vectors[:, order], singular_values[order]

    @staticmethod
    def sigma_is_degenerate(moment_trace: float, mean_energy: float) -> bool:
        """Whether the gradient covariance carries no usable signal."""
        return (
            float(moment_trace) - float(mean_energy)
            <= SIGMA_TRACE_TOL * max(float(mean_energy), 1e-24)
        )

    @torch.no_grad()
    def _mixed_right_eigensystem(
        self,
        matrix: torch.Tensor,
        probe: torch.Tensor,
        moment_sketch: torch.Tensor,
        moment_trace: float,
        mean_energy: float,
        layer_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Nystrom eigensystem of the trace-normalised mixed criterion.

        Selects the dominant eigenspace of

            C_beta = (1 - beta) * GbarT_Gbar / ||Gbar||_F^2
                   + beta * Sigma / tr(Sigma),      Sigma = M - GbarT_Gbar

        which has unit trace for every ``beta``. Both the normalisation and the
        centring matter. Without them the raw second moment carries a trace
        larger than ``||Gbar||_F^2`` by the factor ``snr``, so any
        non-negligible mixing weight is dominated by it and ``beta`` behaves as
        a binary switch rather than a convex interpolation; and the uncentred
        term still contains the mean component that the first term already
        supplies, so the weight double-counts it.

        Neither ``C_beta`` nor ``Sigma`` is ever formed: the centring is one
        subtraction on the sketched side. Returns orthonormal right vectors as
        columns with a singular-value-scaled spectrum, or ``None`` when the
        Nystrom core is numerically rank-deficient. Callers must rule out a
        degenerate ``Sigma`` with :meth:`sigma_is_degenerate` first.
        """
        probe_matrix = probe.to(matrix.device, dtype=torch.float32)
        mean_sketch = matrix.transpose(0, 1) @ (matrix @ probe_matrix)
        trace_sigma = float(moment_trace) - float(mean_energy)
        if trace_sigma <= 0.0:
            raise ValueError(
                "GradLoRA mixed criterion requires a positive tr(Sigma); "
                "call sigma_is_degenerate() before this method"
            )
        sigma_sketch = (
            moment_sketch.to(matrix.device, dtype=torch.float32) - mean_sketch
        )
        mixed_sketch = (
            (1.0 - self.second_moment_beta) / max(float(mean_energy), 1e-24)
        ) * mean_sketch + (self.second_moment_beta / trace_sigma) * sigma_sketch
        eigensystem = self._nystrom_eigh(probe_matrix, mixed_sketch, layer_rank)
        if eigensystem is None:
            return None
        right_vectors, eigenvalues = eigensystem
        return right_vectors, eigenvalues.clamp_min(0).sqrt()

    @torch.no_grad()
    def finalize(
        self,
        *,
        rank: int,
        spectral_exponent: float = 0.0,
        oversample: int = 8,
        niter: int = 4,
        svd_device: str = "cuda",
        init_variant: str = "selected",
        diagnostics_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[GradLoRALayerDiagnostic]:
        """Average gradients, initialize A from top right singular vectors.

        ``init_variant`` swaps the subspace written into A for one of the
        controls the usefulness probe scores, so a frozen-A run can attribute
        reward differences to subspace choice alone. All variants keep the
        row RMS selected by ``a_scale_mode``, so only the subspace differs.
        """
        if init_variant not in {"selected", "mean", "random"}:
            raise ValueError(
                "gradlora init_variant must be 'selected', 'mean', or "
                f"'random', got {init_variant!r}"
            )
        if self.use_second_moment and self.batch_gradients:
            raise RuntimeError(
                "GradLoRA has an unfinished second-moment calibration batch; "
                "call finish_batch() after every calibration backward pass"
            )
        self.close()
        if rank <= 0:
            raise ValueError(f"GradLoRA rank must be positive, got {rank}")
        spectral_exponent = float(spectral_exponent)
        if not math.isfinite(spectral_exponent):
            raise ValueError(
                "gradlora spectral_exponent must be finite, "
                f"got {spectral_exponent}"
            )
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
            (
                gradient,
                total_calls,
                moment_sketch,
                moment_trace,
                probe,
            ) = self._pop_layer_state(name)
            rows, cols = gradient.shape
            layer_rank = min(rank, rows, cols)
            q = min(layer_rank + max(int(oversample), 0), rows, cols)
            gradient_fro_sq = float(gradient.float().pow(2).sum().item())
            gradient_snr = float("nan")
            sigma_degenerate = False
            if self.use_second_moment:
                if (
                    not math.isfinite(moment_trace)
                    or not torch.isfinite(moment_sketch).all()
                ):
                    raise FloatingPointError(
                        f"GradLoRA second moment for {name} contains "
                        "non-finite values"
                    )
                if gradient_fro_sq > 0.0:
                    gradient_snr = moment_trace / gradient_fro_sq
                sigma_degenerate = self.sigma_is_degenerate(
                    moment_trace, gradient_fro_sq
                )
            row_scale, write_gain = _resolve_a_row_scale(
                self.modules[name],
                _active_adapter_name(self.modules[name]),
                self.a_scale_mode,
                self.a_scale,
            )
            init_a: torch.Tensor | None = None
            mean_a: torch.Tensor | None = None
            initialization = (
                "gradient_second_moment"
                if self.use_second_moment
                else "gradient_svd"
            )
            kept_energy = 0.0
            sigma_max = 0.0
            sigma_min_kept = 0.0
            initializer_spectrum: torch.Tensor | None = None

            if process_rank == 0:
                matrix = gradient.to(factor_device, dtype=torch.float32)
                if not torch.isfinite(matrix).all():
                    raise FloatingPointError(
                        f"GradLoRA gradient for {name} contains non-finite values"
                    )
                if gradient_fro_sq <= 0.0:
                    # Some SD3 modules are structurally inactive. In particular,
                    # the final context_pre_only block computes add_q_proj only
                    # for a context output that is discarded, so its exact
                    # gradient is always zero. Keep PEFT's original random A
                    # (and zero B) for such layers instead of aborting all ranks.
                    initialization = "standard_lora_fallback"
                    zero_gradient_layers.append(name)
                else:
                    mixed_eigensystem = None
                    if self.use_second_moment and not sigma_degenerate:
                        mixed_eigensystem = self._mixed_right_eigensystem(
                            matrix,
                            probe,
                            moment_sketch,
                            moment_trace,
                            gradient_fro_sq,
                            layer_rank,
                        )
                    if self.use_second_moment and mixed_eigensystem is not None:
                        right_vectors, initializer_spectrum = mixed_eigensystem
                        init_a = right_vectors.transpose(0, 1).contiguous()
                        # tr(C_beta) = 1 by construction, so the retained
                        # eigenvalues are already the captured fraction. Left
                        # unclamped: a value above one can only come from the
                        # Nystrom sketch, and hiding that would hide the very
                        # approximation error the audit has to detect.
                        kept_energy = float(
                            initializer_spectrum.pow(2).sum().item()
                        )
                        sigma_max = float(initializer_spectrum[0].item())
                        sigma_min_kept = float(
                            initializer_spectrum[layer_rank - 1].item()
                        )
                    else:
                        if self.use_second_moment:
                            # Record which of the two reasons forced beta = 0 on
                            # this layer: a covariance that carries no signal is
                            # a property of the data, a rank-deficient Nystrom
                            # core is a property of the sketch.
                            initialization = (
                                "gradient_svd_sigma_degenerate"
                                if sigma_degenerate
                                else "gradient_svd_nystrom_fallback"
                            )
                        right_vectors, singular_values = (
                            self._mean_right_eigensystem(matrix, q, niter)
                        )
                        init_a = (
                            right_vectors[:, :layer_rank]
                            .transpose(0, 1)
                            .contiguous()
                        )
                        initializer_spectrum = singular_values[:layer_rank]
                        kept_energy = float(
                            singular_values[:layer_rank].pow(2).sum().item()
                            / max(gradient_fro_sq, 1e-24)
                        )
                        sigma_max = float(singular_values[0].item())
                        sigma_min_kept = float(
                            singular_values[layer_rank - 1].item()
                        )
                    if init_a is not None and initializer_spectrum is not None:
                        init_a = _apply_spectral_exponent(
                            init_a,
                            initializer_spectrum,
                            spectral_exponent,
                            row_scale,
                        )
                    if self.use_second_moment:
                        # First-order control: the subspace this layer would
                        # have received with second-moment mixing disabled.
                        # Compute this in every arm, even when it will not be
                        # selected, so randomized SVD consumes the same CUDA RNG
                        # stream before frozen-A training in all three arms.
                        mean_vectors, mean_values = self._mean_right_eigensystem(
                            matrix, q, niter
                        )
                        mean_a = _apply_spectral_exponent(
                            mean_vectors[:, :layer_rank]
                            .transpose(0, 1)
                            .contiguous(),
                            mean_values[:layer_rank],
                            spectral_exponent,
                            row_scale,
                        )
                    if init_variant == "mean" and mean_a is not None:
                        init_a = mean_a
                        initialization = "control_mean"
                        kept_energy = float(
                            mean_values[:layer_rank].pow(2).sum().item()
                            / max(gradient_fro_sq, 1e-24)
                        )
                        sigma_max = float(mean_values[0].item())
                        sigma_min_kept = float(
                            mean_values[layer_rank - 1].item()
                        )
                    elif init_variant == "random":
                        random_a = self._orthonormal_rows(
                            self._random_probe_basis(name, cols, layer_rank),
                            row_scale,
                        ).to(matrix.device)
                        random_basis = random_a / row_scale
                        kept_energy = float(
                            (matrix @ random_basis.transpose(0, 1))
                            .pow(2)
                            .sum()
                            .item()
                            / max(gradient_fro_sq, 1e-24)
                        )
                        sigma_max = 0.0
                        sigma_min_kept = 0.0
                        init_a = random_a.to(init_a.device, init_a.dtype)
                        initialization = "control_random"
            elif gradient_fro_sq > 0.0:
                init_a = torch.empty(
                    layer_rank,
                    cols,
                    device=communication_device,
                    dtype=torch.float32,
                )
            else:
                initialization = "standard_lora_fallback"

            # All ranks agree on gradient_fro_sq after the all-reduce above, so
            # either every rank has an init_a buffer or every rank falls back.
            if distributed and init_a is not None:
                init_a = init_a.to(communication_device)
                dist.broadcast(init_a, src=0)
            if init_a is not None:
                initializers[name] = init_a.cpu()

            if self.collect_probe_state and init_a is not None:
                if distributed and self.use_second_moment:
                    if mean_a is None:
                        mean_a = torch.empty(
                            layer_rank,
                            cols,
                            device=communication_device,
                            dtype=torch.float32,
                        )
                    mean_a = mean_a.to(communication_device)
                    dist.broadcast(mean_a, src=0)
                mean_gradient = gradient.detach().to("cpu", dtype=torch.float32)
                if self.keep_calibration_gradients:
                    self.probe_calibration_gradients[name] = mean_gradient
                # A rows are only orthonormal when spectral_exponent == 0, so
                # re-orthonormalize before using them as a subspace basis.
                self._register_probe_variant("selected", name, init_a, mean_gradient)
                if mean_a is not None:
                    self._register_probe_variant("mean", name, mean_a, mean_gradient)
                self._register_probe_variant(
                    "random",
                    name,
                    self._random_probe_basis(name, cols, layer_rank),
                    mean_gradient,
                )
                complement = self._complement_probe_basis(
                    name,
                    self.probe_subspaces["selected"][name],
                )
                if complement is not None:
                    self._register_probe_variant(
                        "complement",
                        name,
                        complement,
                        mean_gradient,
                    )
                    self._assert_probe_complement(name)

            if process_rank == 0:
                flatness = _spectral_flatness(initializer_spectrum)
                adam_cosine = _first_step_adam_cosine(gradient, init_a)
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
                        a_scale_mode=self.a_scale_mode,
                        write_gain=write_gain,
                        gradient_snr=gradient_snr,
                        spectral_flatness=flatness,
                        predicted_adam_cosine=adam_cosine,
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
                "spectral_exponent": spectral_exponent,
                "oversample": oversample,
                "power_iterations": niter,
                "use_second_moment": self.use_second_moment,
                "second_moment_beta": (
                    self.second_moment_beta
                    if self.use_second_moment
                    else 0.0
                ),
                "a_scale_mode": self.a_scale_mode,
                "a_scale": self.a_scale,
                "layers": [asdict(item) for item in diagnostics],
            }
            if metadata:
                payload.update(metadata)
            write_json_report(diagnostics_path, payload, indent=2)
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

    @staticmethod
    def _orthonormal_rows(
        rows_as_vectors: torch.Tensor,
        target_row_rms: float,
    ) -> torch.Tensor:
        """Orthonormalize rows, then restore PEFT's original A row RMS."""
        result = torch.linalg.qr(
            rows_as_vectors.float().transpose(0, 1),
            mode="reduced",
        ).Q.transpose(0, 1).contiguous()
        current_row_rms = result.norm() / math.sqrt(max(result.shape[0], 1))
        return result * (float(target_row_rms) / float(current_row_rms.item()))

    @staticmethod
    def _probe_seed(name: str, salt: str = "") -> int:
        """Layer-specific seed that every rank reproduces without communicating."""
        digest = hashlib.sha256(f"{name}{salt}".encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big", signed=False) % (2**63)

    @classmethod
    def _random_probe_basis(
        cls,
        name: str,
        cols: int,
        layer_rank: int,
    ) -> torch.Tensor:
        """Draw a layer-specific Haar subspace that every rank reproduces."""
        generator = torch.Generator(device="cpu").manual_seed(cls._probe_seed(name))
        return torch.randn(
            layer_rank,
            cols,
            generator=generator,
            dtype=torch.float32,
        )

    @classmethod
    def _complement_probe_basis(
        cls,
        name: str,
        selected_basis: torch.Tensor,
    ) -> torch.Tensor | None:
        """Seeded random ``r``-frame inside the complement of ``selected_basis``.

        ``selected_basis`` carries the selected subspace as columns. Drawing
        from ``range(I - V V^T)`` gives the only arm that can refute the
        method: a subspace built to avoid the dominant directions entirely. If
        it scores no worse on held-out gradients, the rank budget is being
        spent in the wrong place and energy capture is the wrong objective.

        Returns ``None`` when the complement is too narrow to hold ``r``
        orthonormal directions, which cannot happen at the ranks used here.
        """
        cols, layer_rank = selected_basis.shape
        if cols - layer_rank < layer_rank:
            return None
        generator = torch.Generator(device="cpu").manual_seed(
            cls._probe_seed(name, ":complement")
        )
        draw = torch.randn(
            cols,
            layer_rank,
            generator=generator,
            dtype=torch.float32,
        )
        basis = selected_basis.detach().float().cpu()
        projected = draw - basis @ (basis.transpose(0, 1) @ draw)
        complement = torch.linalg.qr(projected, mode="reduced").Q
        return complement.transpose(0, 1).contiguous()

    def _assert_probe_complement(self, name: str, tolerance: float = 1e-5) -> None:
        """Verify the stored complement basis is orthonormal and disjoint."""
        selected = self.probe_subspaces["selected"][name].float()
        complement = self.probe_subspaces["complement"][name].float()
        overlap = float(
            (selected.transpose(0, 1) @ complement).norm().item()
        )
        gram_error = float(
            (
                complement.transpose(0, 1) @ complement
                - torch.eye(complement.shape[1], dtype=complement.dtype)
            )
            .norm()
            .item()
        )
        if overlap >= tolerance or gram_error >= tolerance:
            raise RuntimeError(
                f"GradLoRA complement probe basis for {name} is invalid: "
                f"||V^T Q||_F={overlap:.3e}, ||Q^T Q - I||_F={gram_error:.3e}"
            )

    def _register_probe_variant(
        self,
        variant: str,
        name: str,
        rows_as_vectors: torch.Tensor,
        mean_gradient: torch.Tensor,
    ) -> None:
        """Store an orthonormal basis and its calibration-gradient projection."""
        basis = torch.linalg.qr(
            rows_as_vectors.detach().transpose(0, 1).float().cpu(),
            mode="reduced",
        ).Q
        reference = mean_gradient @ basis
        self.probe_subspaces.setdefault(variant, {})[name] = basis
        self.probe_references.setdefault(variant, {})[name] = reference
        self.probe_reference_energy.setdefault(variant, {})[name] = float(
            reference.pow(2).sum().item()
        )

    def begin_probe(self) -> None:
        """Re-open hooks to score held-out gradients against the chosen subspaces."""
        if not self.collect_probe_state:
            raise RuntimeError(
                "GradLoRA usefulness probing requires collect_probe_state=True"
            )
        if not self.probe_subspaces:
            raise RuntimeError(
                "GradLoRA usefulness probing found no calibrated subspace; "
                "call finalize() first"
            )
        if self.handles:
            raise RuntimeError("GradLoRA probe hooks are already registered")

        self.probe_mode = True
        self.gradients.clear()
        self.batch_gradients.clear()
        self.probe_layer_stats = {
            variant: {
                name: {
                    "held_out_energy": 0.0,
                    "captured_energy": 0.0,
                    "reference_dot": 0.0,
                }
                for name in layers
            }
            for variant, layers in self.probe_subspaces.items()
        }
        self.probe_batch_records = {
            variant: [] for variant in self.probe_subspaces
        }
        main_print(
            "--> GradLoRA usefulness probe variants: "
            + ", ".join(sorted(self.probe_subspaces))
        )
        self._register_hooks(phase="usefulness-probe")

    @staticmethod
    def _reduce_sum(values: list[float]) -> tuple[list[float], int]:
        """All-reduce a small vector of scalars and report the world size."""
        if not (dist.is_available() and dist.is_initialized()):
            return list(values), 1
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        packed = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return [float(item) for item in packed.tolist()], dist.get_world_size()

    @torch.no_grad()
    def _reduced_batch_gradients(self) -> dict[str, torch.Tensor]:
        """Data-parallel mean of this batch's per-layer virtual gradients.

        Used by both calibration and the held-out probe, which is the point:
        one batch means one *global* batch on both sides. Any per-rank
        alternative silently redefines the batch, and every quantity that is a
        statement about between-batch spread -- ``snr``, ``Sigma``, the beta
        mixture, the held-out/calibration comparison -- would then be read
        against the wrong noise level.

        Layers are iterated in a fixed order and missing ones contributed as
        zeros, because a layer can be inactive on one rank and active on
        another, and the ranks must agree on the collective schedule.
        """
        local = {
            name: tensor.to(self.accumulation_device, dtype=torch.float32)
            for name, tensor in self.batch_gradients.items()
        }
        if not (dist.is_available() and dist.is_initialized()):
            return local

        backend = dist.get_backend()
        reduce_device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl"
            else torch.device("cpu")
        )
        world_size = dist.get_world_size()
        reduced: dict[str, torch.Tensor] = {}
        for name in sorted(self.modules):
            module = self.modules[name]
            shape = (
                int(module.base_layer.out_features),
                int(module.base_layer.in_features),
            )
            tensor = local.get(name)
            packed = (
                torch.zeros(shape, dtype=torch.float32, device=reduce_device)
                if tensor is None
                else tensor.to(reduce_device)
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            packed /= world_size
            reduced[name] = packed.to(self.accumulation_device)
        return reduced

    @torch.no_grad()
    def _finish_probe_batch(self) -> None:
        """Score one held-out batch against the calibrated subspaces.

        A rank that captured nothing must still run every collective below
        with a zero contribution. Returning early instead would leave the
        remaining ranks waiting in an all-reduce that never completes, and a
        ReLU-clipped reward makes an empty rank a routine occurrence rather
        than an edge case.
        """
        if not self.batch_gradients:
            main_print(
                "[warning] GradLoRA usefulness probe captured no layer "
                "gradients on this rank; contributing zeros to this held-out "
                "batch"
            )

        if self.probe_reduce_gradients:
            reduced = self._reduced_batch_gradients()
            if self.probe_observer is not None:
                self.probe_observer.observe_batch(
                    self._probe_batch_index,
                    reduced,
                )
            self._probe_batch_index += 1

        # The legacy per-variant summary below stays on this rank's own shard,
        # so it treats each shard as a sample and reads at a different noise
        # level than the RQ3 report, which scores global batches. Both are
        # internally consistent; the RQ3 report is the one the write-up quotes.
        device = self.moment_compute_device
        gradients = {
            name: tensor.to(device, dtype=torch.float32)
            for name, tensor in self.batch_gradients.items()
        }
        for variant, subspaces in self.probe_subspaces.items():
            held_out_energy = 0.0
            captured_energy = 0.0
            reference_dot = 0.0
            for name, gradient in gradients.items():
                basis = subspaces.get(name)
                if basis is None:
                    continue
                projection = gradient @ basis.to(device, dtype=torch.float32)
                reference = self.probe_references[variant][name].to(
                    device,
                    dtype=torch.float32,
                )

                layer_energy = float(gradient.pow(2).sum().item())
                layer_captured = float(projection.pow(2).sum().item())
                layer_dot = float((projection * reference).sum().item())

                stats = self.probe_layer_stats[variant][name]
                stats["held_out_energy"] += layer_energy
                stats["captured_energy"] += layer_captured
                stats["reference_dot"] += layer_dot

                held_out_energy += layer_energy
                captured_energy += layer_captured
                reference_dot += layer_dot

            # Per-rank cosine is exact, so average it instead of ratios of sums.
            direction_norm = math.sqrt(
                sum(self.probe_reference_energy[variant].values())
            )
            cosine_defined = direction_norm > 0.0 and held_out_energy > 0.0
            # A NaN from an empty rank would propagate through the sum and
            # destroy the cosine for every rank, so undefined ranks contribute
            # zero and are excluded from the denominator instead.
            local_cosine = (
                reference_dot / (direction_norm * math.sqrt(held_out_energy))
                if cosine_defined
                else 0.0
            )
            reduced, _ = self._reduce_sum(
                [
                    held_out_energy,
                    captured_energy,
                    reference_dot,
                    local_cosine,
                    1.0 if cosine_defined else 0.0,
                ]
            )
            total_energy, total_captured, total_dot, cosine_sum = reduced[:4]
            cosine_samples = int(round(reduced[4]))
            self.probe_batch_records[variant].append(
                {
                    "held_out_energy": total_energy,
                    "captured_energy": total_captured,
                    "reference_dot": total_dot,
                    "capture_fraction": (
                        total_captured / total_energy
                        if total_energy > 0.0
                        else float("nan")
                    ),
                    "alignment_cosine": (
                        cosine_sum / cosine_samples
                        if cosine_samples > 0
                        else float("nan")
                    ),
                    "samples": cosine_samples,
                }
            )
        self.batch_gradients.clear()

    @torch.no_grad()
    def finish_probe(
        self,
        *,
        diagnostics_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Close probe hooks and summarize held-out capture and usefulness."""
        self.close()
        self.probe_mode = False
        if not any(self.probe_batch_records.values()):
            raise RuntimeError(
                "GradLoRA usefulness probe scored no held-out batch"
            )

        variants: dict[str, Any] = {}
        for variant, layer_stats in self.probe_layer_stats.items():
            records = self.probe_batch_records[variant]
            names = sorted(layer_stats)
            packed = [
                layer_stats[name][key]
                for name in names
                for key in (
                    "held_out_energy",
                    "captured_energy",
                    "reference_dot",
                )
            ]
            reduced, _ = self._reduce_sum(packed)
            layer_rows: list[dict[str, Any]] = []
            for index, name in enumerate(names):
                energy, captured, dot = reduced[3 * index : 3 * index + 3]
                layer_rows.append(
                    {
                        "name": name,
                        "held_out_energy": energy,
                        "captured_energy": captured,
                        "capture_fraction": (
                            captured / energy if energy > 0.0 else float("nan")
                        ),
                        "reference_dot": dot,
                        "reference_energy": (
                            self.probe_reference_energy[variant][name]
                        ),
                    }
                )

            total_energy = sum(row["held_out_energy"] for row in layer_rows)
            total_captured = sum(row["captured_energy"] for row in layer_rows)
            total_dot = sum(row["reference_dot"] for row in layer_rows)
            total_samples = sum(record["samples"] for record in records)
            direction_norm = math.sqrt(
                sum(self.probe_reference_energy[variant].values())
            )
            variants[variant] = {
                "selection_direction_norm": direction_norm,
                "aggregate": {
                    "capture_fraction": (
                        total_captured / total_energy
                        if total_energy > 0.0
                        else float("nan")
                    ),
                    "alignment_cosine": (
                        sum(record["alignment_cosine"] for record in records)
                        / len(records)
                    ),
                    "projected_gain": (
                        total_dot / (total_samples * direction_norm)
                        if direction_norm > 0.0 and total_samples > 0
                        else float("nan")
                    ),
                },
                "per_batch": records,
                "layers": layer_rows,
            }

        reference_variant = next(iter(self.probe_batch_records))
        summary: dict[str, Any] = {
            "method": "GradLoRA usefulness probe",
            "probe_batches": len(self.probe_batch_records[reference_variant]),
            "held_out_samples": sum(
                record["samples"]
                for record in self.probe_batch_records[reference_variant]
            ),
            "variants": variants,
            "definitions": {
                "variants": (
                    "'selected' is the subspace this run actually initialized "
                    "LoRA A with; 'mean' is the first-order control obtained "
                    "from the mean calibration gradient alone; 'random' is a "
                    "Haar control of identical rank. All variants are scored "
                    "on the same held-out batches."
                ),
                "capture_fraction": (
                    "Energy-weighted ||G_heldout V||_F^2 / ||G_heldout||_F^2 "
                    "over all probed layers. Measures how much held-out "
                    "gradient energy the subspace retains."
                ),
                "alignment_cosine": (
                    "Mean over held-out samples of <G_heldout, D> / "
                    "(||G_heldout||_F ||D||_F), where D = G_calibration V V^T "
                    "is the update direction implied by the subspace."
                ),
                "projected_gain": (
                    "Mean <G_heldout, D> / ||D||_F. Because the FlowBP loss is "
                    "-reward and the update is -eta D, a positive value "
                    "predicts a held-out reward increase."
                ),
            },
        }
        if metadata:
            summary.update(metadata)

        process_rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        if process_rank == 0 and diagnostics_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(diagnostics_path)),
                exist_ok=True,
            )
            write_json_report(diagnostics_path, summary, indent=2)
            main_print(f"--> GradLoRA usefulness saved to {diagnostics_path}")

        for variant in sorted(variants):
            aggregate = variants[variant]["aggregate"]
            main_print(
                f"--> GradLoRA held-out [{variant:>8}] "
                f"capture={aggregate['capture_fraction']:.4f}, "
                f"alignment cosine={aggregate['alignment_cosine']:.4f}, "
                f"projected gain={aggregate['projected_gain']:.6f}"
            )
        return summary
