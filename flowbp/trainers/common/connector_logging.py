"""Connector diagnostics shared by every FlowBP backbone.

Saving a connector dump and logging decoded connector pairs to W&B are both
backend-agnostic: the only part that differs is how latents are turned into
images, which callers inject as ``decode_fn``.

These helpers previously existed as eight near-identical copies across the
FLUX.1, FLUX.2 and SD3.5 trainers. The copies had drifted on one point: only
the SD3.5 one fell back to ``connector_dump_interval`` when
``connector_wandb_interval`` was unset. That (more useful) behaviour is the one
kept here.

Per-sample trajectory filtering is deliberately *not* shared: SD3.5 applies a
boolean keep-mask while FLUX.2 folds a soft weight into the similarity weights,
so the two are not interchangeable.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.distributed as dist
import wandb

CONNECTOR_IMAGE_KEYS = ("xj_pred", "xj", "x0_pred", "x0")
ENDPOINT_IMAGE_KEYS = ("x0_pred", "x0_true")


def _is_logging_rank() -> bool:
    return not (
        dist.is_available() and dist.is_initialized() and dist.get_rank() != 0
    )


def _due(args, interval_attr: str, fallback_attr: str | None = None) -> int | None:
    """Return the current step when ``interval_attr`` fires on it, else None."""
    raw_interval = getattr(args, interval_attr, None)
    if raw_interval is None and fallback_attr is not None:
        raw_interval = getattr(args, fallback_attr, 0)
    interval = int(raw_interval or 0)
    step = getattr(args, "current_train_step", None)
    if interval <= 0 or step is None or int(step) % interval != 0:
        return None
    return int(step)


def maybe_save_connector_dump(args, prefix: str, payload: dict[str, Any]) -> None:
    """Persist a connector payload every ``connector_dump_interval`` steps."""
    step = _due(args, "connector_dump_interval")
    if step is None or not _is_logging_rank():
        return

    dump_dir = getattr(args, "connector_dump_dir", None) or os.path.join(
        args.output_dir, "connector_dumps"
    )
    os.makedirs(dump_dir, exist_ok=True)
    saved = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in payload.items()
    }
    torch.save(saved, os.path.join(dump_dir, f"{prefix}_step_{step:06d}.pt"))


def maybe_log_connector_wandb(
    args,
    prefix: str,
    payload: dict[str, Any] | None,
    decode_fn: Callable[[torch.Tensor, int], list],
    keys: tuple[str, ...] = CONNECTOR_IMAGE_KEYS,
) -> None:
    """Log decoded connector latents to W&B.

    ``decode_fn(latents, max_samples)`` returns a list of HWC float arrays; it
    owns the backend-specific unpack plus VAE denormalization. ``keys`` differs
    per variant: the bridged trainers carry an ``xj`` pair, while FlowBP-Sparse
    only has an endpoint pair named ``x0_pred`` / ``x0_true``.
    """
    step = _due(args, "connector_wandb_interval", "connector_dump_interval")
    if step is None or not _is_logging_rank():
        return
    if payload is None or wandb.run is None:
        return

    max_samples = int(getattr(args, "connector_wandb_num_samples", 2) or 2)
    log_payload = {}
    for name in keys:
        latents = payload.get(name)
        if latents is None:
            continue
        images = decode_fn(latents, max_samples)
        if images:
            log_payload[f"connector/{prefix}_{name}"] = [
                wandb.Image(image, caption=f"{prefix} {name} sample {idx}")
                for idx, image in enumerate(images)
            ]
    if log_payload:
        wandb.log(log_payload, step=step)


def images_to_wandb_arrays(images: torch.Tensor) -> list:
    """Convert a decoded CHW image batch into the HWC arrays W&B expects."""
    return [
        image.detach().float().cpu().permute(1, 2, 0).numpy() for image in images
    ]
