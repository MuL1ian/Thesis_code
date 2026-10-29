"""Backend defaults and legacy argument spellings.

``flowbp/config/parser.py`` builds the argument parser. ``BACKEND_DEFAULTS``
re-points the arguments whose default is set per backbone. SD3.5 keeps the
values the training runs were made with, so existing configs resolve to the
same arguments.
"""

from __future__ import annotations

from typing import Any

_SD35: dict[str, Any] = {
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
    "project": "flowbp",
    "run_name": "flowbp_sd35",
    "sampling_steps": None,
    "train_batch_size": 16,
    "train_sp_batch_size": 1,
    "w": None,
}

BACKEND_DEFAULTS: dict[str, dict[str, Any]] = {
    "sd35": dict(_SD35),
}

SUPPORTED_BACKENDS = tuple(BACKEND_DEFAULTS)

# Legacy CLI/YAML spellings mapped onto the canonical destination. The
# misspelled ``fsdp_sharding_startegy`` stays canonical because every trainer
# reads that attribute; the corrected spelling is accepted as an alias.
ARG_ALIASES: dict[str, str] = {
    "fsdp_sharding_strategy": "fsdp_sharding_startegy",
}
