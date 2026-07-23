"""Shared dispatch for differentiable FlowBP reward models."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


SUPPORTED_REWARD_MODELS = ("hpsv2", "mps", "hpsv3plusplus")


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


def init_mps_reward(args, device):
    from flowbp.rewards.mps_reward import MPSDifferentiableReward

    model = MPSDifferentiableReward(
        device=torch.device("cuda", device) if isinstance(device, int) else device,
        checkpoint_path=getattr(args, "reward_ckpt_path", None),
        processor_name=getattr(args, "mps_processor_name", None),
        condition_text=getattr(args, "mps_condition", None),
        mask_threshold=float(getattr(args, "mps_mask_threshold", 0.3)),
    )
    return model, None, None


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
    if model_type == "mps":
        return reward_model(images, prompt_list)

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
