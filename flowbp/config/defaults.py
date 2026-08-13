"""Per-backend defaults and legacy argument spellings.

``flowbp/config/parser.py`` builds one argument parser covering every backend.
The parser's own ``default=`` values are the FLUX.1 / SD3.5 defaults, because
those two shared a parser historically; ``BACKEND_DEFAULTS`` then re-points the
arguments whose default genuinely differs per backbone.

Only arguments that actually differed between the old ``train_flowbp_flux.py``
and ``train_flowbp_flux2.py`` parsers appear here. SD3.5 intentionally carries
the FLUX.1 values because ``train_flowbp_sd35.py`` used to import the FLUX.1
parser directly, so anything else would silently change existing SD3.5 runs.
"""

from __future__ import annotations

from typing import Any

# FLUX.1 values live in the parser itself; listed here only so the three
# backends can be compared side by side in one place.
_FLUX1: dict[str, Any] = {
    "cfg_guidance": 3.5,
    "dataloader_num_workers": 10,
    "eval_num_imgs_per_prompt": 4,
    "evaluation_interval": 500,
    "flowbp_lagrange_connector_order": 4,
    "flowbp_lagrange_grad_rescale": 0.0,
    "flowbp_lagrange_grad_support_mode": "start",
    "flowbp_lagrange_grad_support_scale": 1.0,
    "flowbp_lagrange_max_active_supports": 0,
    "h": None,
    "learning_rate": 1e-4,
    "loss_relu_clip": 0.7,
    "master_weight_type": "fp32",
    "max_grad_norm": 2.0,
    "max_idx": 10,
    "pretrained_model_name_or_path": None,
    "project": "flux_rlhf",
    "run_name": "flux_grpo",
    "sampling_steps": None,
    "train_batch_size": 16,
    "train_sp_batch_size": 1,
    "w": None,
}

_FLUX2: dict[str, Any] = {
    "cfg_guidance": 4.0,
    "dataloader_num_workers": 4,
    "eval_num_imgs_per_prompt": 1,
    "evaluation_interval": 0,
    "flowbp_lagrange_connector_order": 3,
    "flowbp_lagrange_grad_rescale": 0.5,
    "flowbp_lagrange_grad_support_mode": "midpoint",
    "flowbp_lagrange_grad_support_scale": 0.25,
    "flowbp_lagrange_max_active_supports": 2,
    "h": 512,
    "learning_rate": 1e-5,
    "loss_relu_clip": 0.55,
    "master_weight_type": "bf16",
    "max_grad_norm": 1.0,
    "max_idx": 26,
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
    "project": "flux2_rlhf",
    "run_name": "flux2_leapalign",
    "sampling_steps": 25,
    "train_batch_size": 4,
    "train_sp_batch_size": 4,
    "w": 512,
}

BACKEND_DEFAULTS: dict[str, dict[str, Any]] = {
    "flux1": dict(_FLUX1),
    "flux2": dict(_FLUX2),
    "sd35": dict(_FLUX1),
}

SUPPORTED_BACKENDS = tuple(BACKEND_DEFAULTS)

# Legacy CLI/YAML spellings mapped onto the canonical destination. The
# misspelled ``fsdp_sharding_startegy`` stays canonical because every trainer
# reads that attribute; the corrected spelling is accepted as an alias.
ARG_ALIASES: dict[str, str] = {
    "fsdp_sharding_strategy": "fsdp_sharding_startegy",
}
