"""Shared dispatch for differentiable FlowBP reward models."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F


SUPPORTED_REWARD_MODELS = ("hpsv2", "hpsv3plusplus")

_REPO_ROOT = Path(__file__).resolve().parents[2]

HPSV2_BACKBONE_FILENAMES = ("open_clip_pytorch_model.bin",)
HPSV2_REWARD_FILENAMES = ("HPS_v2.1_compressed.pt", "HPS_v2_compressed.pt")


def reward_model_type(args) -> str:
    name = str(getattr(args, "reward_model_type", "") or "").strip().lower()
    if not name:
        name = "hpsv2" if bool(getattr(args, "use_hpsv2", False)) else ""
    if name not in SUPPORTED_REWARD_MODELS:
        raise ValueError(
            f"Unsupported reward_model_type={name!r}; "
            f"expected one of {SUPPORTED_REWARD_MODELS}."
        )
    return name


def normalize_prompts(prompts, batch_size: int) -> list[str]:
    if isinstance(prompts, str):
        return [prompts] * batch_size
    if not isinstance(prompts, Iterable):
        raise TypeError(f"Unsupported prompt batch type: {type(prompts)}")
    result = [str(prompt) for prompt in prompts]
    if len(result) != batch_size:
        raise ValueError(f"Prompt batch mismatch: {len(result)} != {batch_size}")
    return result


def reward_ckpt_search_dirs(args) -> list[str]:
    """Ordered, de-duplicated list of directories to probe for reward weights."""
    candidates: list[str] = []

    def add(value) -> None:
        if not value:
            return
        path = os.path.expanduser(str(value).strip())
        if os.path.isfile(path):
            path = os.path.dirname(path) or "."
        if path and path not in candidates:
            candidates.append(path)

    add(getattr(args, "reward_ckpt_path", None))
    add(getattr(args, "eval_reward_ckpt_path", None))
    add(os.environ.get("FLOWBP_REWARD_CKPT_PATH", ""))
    add(_REPO_ROOT / "models" / "reward_ckpts")
    add(_REPO_ROOT / "reward_ckpts")
    add("./hps_ckpt")
    add(_REPO_ROOT / "hps_ckpt")
    return candidates


def _find_first(search_dirs: list[str], filenames: tuple[str, ...]) -> str | None:
    for directory in search_dirs:
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                return path
    return None


def resolve_hpsv2_checkpoints(args) -> tuple[str, str]:
    """Locate the OpenCLIP backbone and the HPSv2 reward checkpoint.

    Each file is probed independently across the whole search path, so a
    partially populated directory still resolves. Falls back to the downloading
    resolver in ``flowbp.utils.hps_ckpt`` when nothing is present locally.
    """
    explicit_path = os.path.expanduser(
        str(getattr(args, "reward_ckpt_path", "") or "").strip()
    )
    explicit_is_file = bool(explicit_path and os.path.isfile(explicit_path))
    explicit_basename = os.path.basename(explicit_path)

    search_dirs = reward_ckpt_search_dirs(args)
    backbone_path = (
        explicit_path
        if explicit_is_file
        and explicit_basename in HPSV2_BACKBONE_FILENAMES
        else _find_first(search_dirs, HPSV2_BACKBONE_FILENAMES)
    )
    reward_path = (
        explicit_path
        if explicit_is_file
        and explicit_basename not in HPSV2_BACKBONE_FILENAMES
        else _find_first(search_dirs, HPSV2_REWARD_FILENAMES)
    )
    if backbone_path and reward_path:
        return backbone_path, reward_path

    from flowbp.utils.hps_ckpt import ensure_hpsv2_ckpts

    return ensure_hpsv2_ckpts()


def init_hpsv2_reward(args, device):
    """Build the frozen HPSv2.1 reward model used as the training objective.

    Returns ``(reward_model, tokenizer, preprocess_val)`` where
    ``preprocess_val`` is the gradient-preserving image transform.
    """
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    from flowbp.utils.hpsv2_transforms import HPSV2TransformsWithGrad

    backbone_path, reward_path = resolve_hpsv2_checkpoints(args)

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        backbone_path,
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    preprocess_val = HPSV2TransformsWithGrad(preprocess_val)

    map_location = f"cuda:{device}" if isinstance(device, int) else device
    checkpoint = torch.load(reward_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")

    reward_model = model.to(device)
    reward_model.requires_grad_(False)
    reward_model.eval()
    return reward_model, tokenizer, preprocess_val


def init_differentiable_reward(args, device):
    """Build the training reward for ``args.reward_model_type``.

    Returns ``(reward_model, processor, preprocess_val)``. HPSv3++ carries its
    own preprocessing, so both trailing values are ``None`` for it.
    """
    selected = reward_model_type(args)
    if selected == "hpsv2":
        return init_hpsv2_reward(args, device)
    return init_hpsv3plusplus_reward(args, device)


def init_hpsv3plusplus_reward(args, device):
    # Importing the scorer module applies the transformers VideoInput patch
    # before any ``hpsv3`` package import.
    from flowbp.eval.rewards.hpsv3plusplus_scorer import (
        HPSv3PlusPlusInferencer,
        discover_hpsv3plusplus_paths,
    )

    config_path, checkpoint_path = discover_hpsv3plusplus_paths(
        reward_ckpt_path=getattr(args, "eval_reward_ckpt_path", "")
        or getattr(args, "reward_ckpt_path", ""),
        config_path=getattr(args, "eval_hpsv3_config_path", ""),
        checkpoint_path=getattr(args, "eval_hpsv3_checkpoint_path", ""),
    )
    if not config_path or not checkpoint_path:
        raise FileNotFoundError(
            "HPSv3++ training requires local config and checkpoint paths. "
            "Set eval.hpsv3_config_path and eval.hpsv3_checkpoint_path in the YAML."
        )

    device_str = f"cuda:{int(device)}" if isinstance(device, int) else str(device)
    reward_model = HPSv3PlusPlusInferencer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device_str,
        differentiable=True,
    )
    reward_model.model.eval()
    reward_model.model.requires_grad_(False)
    return reward_model, None, None


def _as_chw_reward_tensors(images: torch.Tensor) -> list[torch.Tensor]:
    """Convert decoded NCHW [0, 1] images to per-sample CHW tensors."""

    images = images.float().clamp(0, 1)
    return [image.contiguous() for image in images]


def resolve_hpsv3plusplus_iter_step(args, device, batch_size: int) -> torch.Tensor:
    """Linearly ramp HPSv3++ RL-iteration condition from start to end."""

    current_step = float(getattr(args, "current_train_step", 1) or 1)
    max_steps = max(float(getattr(args, "max_train_steps", 1) or 1), 1.0)
    if max_steps <= 1:
        progress = 1.0
    else:
        progress = min(max((current_step - 1.0) / (max_steps - 1.0), 0.0), 1.0)
    iter_start = float(getattr(args, "hpsv3plusplus_iter_start", 0.3))
    iter_end = float(getattr(args, "hpsv3plusplus_iter_end", 1.0))
    iter_value = iter_start + (iter_end - iter_start) * progress
    return torch.full(
        (int(batch_size),),
        float(iter_value),
        device=device,
        dtype=torch.float32,
    )


def compute_differentiable_reward(
    args,
    images: torch.Tensor,
    prompts,
    reward_model,
    tokenizer=None,
    preprocess=None,
) -> torch.Tensor:
    """Return one differentiable reward per image."""

    model_type = reward_model_type(args)
    prompt_list = normalize_prompts(prompts, images.shape[0])
    if model_type == "hpsv3plusplus":
        images_for_reward = _as_chw_reward_tensors(images)
        iter_step = resolve_hpsv3plusplus_iter_step(
            args,
            images.device,
            len(images_for_reward),
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            reward = reward_model.reward(
                images_for_reward,
                prompt_list,
                iter_step=iter_step,
            )
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, device=images.device, dtype=torch.float32)
        reward = reward.float()
        if reward.ndim == 2 and reward.shape[1] >= 1:
            reward = reward[:, 0]
        elif reward.ndim > 1:
            reward = reward.reshape(reward.shape[0], -1)[:, 0]
        return reward

    if tokenizer is None or preprocess is None:
        raise RuntimeError("HPSv2 reward requires tokenizer and image preprocessing.")
    image_input = preprocess(images)
    text = tokenizer(prompt_list).to(
        device=image_input.device,
        non_blocking=True,
    )
    with torch.amp.autocast("cuda", enabled=image_input.is_cuda):
        outputs = reward_model(image_input, text)
        return torch.einsum(
            "bc,bc->b",
            outputs["image_features"],
            outputs["text_features"],
        )


def reward_loss_values(args, rewards: torch.Tensor) -> torch.Tensor:
    """Per-sample maximization loss with optional hinge / direct negation."""

    loss_type = str(getattr(args, "reward_loss_type", "hinge") or "hinge").lower()
    scale = float(getattr(args, "loss_grad_scale", 1.0))
    if loss_type == "neg_reward":
        return -rewards * scale
    if loss_type in {"hinge", "relu_hinge"}:
        clip = float(getattr(args, "loss_relu_clip", 0.0))
        if clip > 0:
            return F.relu(-rewards + clip) * scale
        return -rewards * scale
    raise ValueError(f"Unsupported reward_loss_type: {loss_type}")
