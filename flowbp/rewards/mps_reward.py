"""Differentiable Multi-dimensional Preference Score (MPS) reward."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from flowbp.rewards.mps_compat import (
    ensure_mps_compat_modules,
    normalize_hf_clip_configs,
)


DEFAULT_MPS_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
DEFAULT_MPS_CONDITION = (
    "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, "
    "hands, limbs, structure, instance, texture, quantity, attributes, position, "
    "number, location, word, things."
)
DEFAULT_MPS_MASK_THRESHOLD = 0.3


def find_mps_processor(
    checkpoint_path: str | os.PathLike | None,
    requested: str | os.PathLike | None = None,
) -> str:
    if requested:
        return str(requested)
    env_processor = os.environ.get("FLOWBP_MPS_PROCESSOR", "").strip()
    if env_processor:
        return env_processor

    checkpoint_root = Path(checkpoint_path).expanduser() if checkpoint_path else None
    roots = []
    if checkpoint_root is not None:
        roots.append(checkpoint_root if checkpoint_root.is_dir() else checkpoint_root.parent)
        roots.append(roots[0].parent)
    cache_name = "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    for root in roots:
        snapshots = root / cache_name / "snapshots"
        if snapshots.is_dir():
            revisions = sorted(path for path in snapshots.iterdir() if path.is_dir())
            if revisions:
                return str(revisions[-1])
    return DEFAULT_MPS_PROCESSOR


def find_mps_checkpoint(path: str | os.PathLike | None = None) -> str:
    """Resolve an MPS checkpoint from a file, reward root, or environment."""

    requested = (
        os.environ.get("FLOWBP_MPS_MODEL_PATH", "").strip()
        or str(path or "").strip()
    )
    if not requested:
        workspace_root = Path(__file__).resolve().parents[2]
        requested = str(workspace_root / "models" / "reward_ckpts")

    root = Path(requested).expanduser()
    candidates = (
        [root]
        if root.is_file()
        else [
            root / "mps" / "MPS_overall_checkpoint.pth",
            root / "MPS_overall_checkpoint.pth",
            root / "outputs" / "MPS_overall_checkpoint.pth",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Unable to find MPS_overall_checkpoint.pth. Set reward.ckpt_path, "
        "pass --reward_ckpt_path, or set FLOWBP_MPS_MODEL_PATH.\n"
        f"Searched:\n{searched}"
    )


def _load_mps_model(path: str) -> nn.Module:
    ensure_mps_compat_modules()
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    loaded = normalize_hf_clip_configs(loaded)
    if isinstance(loaded, nn.Module):
        return loaded
    if isinstance(loaded, dict):
        for key in ("model", "module"):
            if isinstance(loaded.get(key), nn.Module):
                return loaded[key]
    raise ValueError(
        "Unsupported MPS checkpoint format: expected nn.Module or a dict "
        "containing an nn.Module under 'model'/'module'."
    )


def _as_hw(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    if isinstance(value, dict):
        height = value.get("height", value.get("shortest_edge"))
        width = value.get("width", value.get("shortest_edge", height))
        if height is not None and width is not None:
            return int(height), int(width)
    raise TypeError(f"Cannot infer MPS processor image size from {value!r}")


class MPSTensorTransform(nn.Module):
    """CLIP preprocessing that preserves gradients to NCHW images in [0, 1]."""

    def __init__(self, image_processor):
        super().__init__()
        target = getattr(image_processor, "crop_size", None) or image_processor.size
        self.height, self.width = _as_hw(target)
        self.register_buffer(
            "mean",
            torch.tensor(image_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(image_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"MPS expects NCHW RGB images, got shape={tuple(images.shape)}"
            )
        images = F.interpolate(
            images.float().clamp(0, 1),
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = self.mean.to(device=images.device, dtype=images.dtype)
        std = self.std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std


class MPSDifferentiableReward(nn.Module):
    """Frozen overall-preference MPS model with image-input autograd enabled."""

    def __init__(
        self,
        device: torch.device | str,
        checkpoint_path: str | os.PathLike | None = None,
        processor_name: str | None = None,
        condition_text: str | None = None,
        mask_threshold: float = DEFAULT_MPS_MASK_THRESHOLD,
    ):
        super().__init__()
        from transformers import AutoTokenizer, CLIPImageProcessor

        self.device = torch.device(device)
        self.processor_name = find_mps_processor(checkpoint_path, processor_name)
        self.condition_text = condition_text or DEFAULT_MPS_CONDITION
        self.mask_threshold = float(mask_threshold)

        image_processor = CLIPImageProcessor.from_pretrained(self.processor_name)
        self.preprocess = MPSTensorTransform(image_processor)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.processor_name,
            trust_remote_code=True,
        )
        self.model_path = find_mps_checkpoint(checkpoint_path)
        self.model = _load_mps_model(self.model_path).eval().to(self.device)
        required = ("model", "cross_model", "logit_scale")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise ValueError(f"MPS checkpoint is missing required attributes: {missing}")
        self.model.requires_grad_(False)
        self.eval()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        target = kwargs.get("device")
        if target is None and args and isinstance(args[0], (str, int, torch.device)):
            target = args[0]
        if target is not None:
            self.device = (
                torch.device("cuda", target)
                if isinstance(target, int)
                else torch.device(target)
            )
        return result

    def _tokenize(self, captions: Iterable[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            list(captions),
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens.input_ids.to(self.device, dtype=torch.long, non_blocking=True)

    def forward(self, images: torch.Tensor, prompts: Iterable[str]) -> torch.Tensor:
        prompts = list(prompts)
        if images.shape[0] != len(prompts):
            raise ValueError(
                f"MPS image/prompt batch mismatch: {images.shape[0]} != {len(prompts)}"
            )
        if not prompts:
            return images.new_empty((0,), dtype=torch.float32)

        image_input = self.preprocess(images).to(self.device)
        text_input = self._tokenize(prompts)
        condition_input = self._tokenize([self.condition_text]).repeat(len(prompts), 1)

        # Text and the hard condition mask are constants. The image branch below
        # intentionally remains outside no_grad so reward gradients reach the VAE.
        with torch.no_grad():
            text_f, text_features = self.model.model.get_text_features(text_input)
            condition_f, _ = self.model.model.get_text_features(condition_input)
            similarities = einsum("b i d, b j d -> b j i", text_f, condition_f)
            similarities = similarities.max(dim=1, keepdim=True)[0]
            similarities = similarities / similarities.max().clamp_min(1e-6)
            mask = torch.where(
                similarities > self.mask_threshold,
                0,
                float("-inf"),
            )

        with torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            image_f = self.model.model.get_image_features(image_input)

        mask = mask.repeat(1, image_f.shape[1], 1)
        cross_dtype = next(self.model.cross_model.parameters()).dtype
        image_f = image_f.to(dtype=cross_dtype)
        text_f = text_f.to(device=image_f.device, dtype=cross_dtype)
        mask = mask.to(device=image_f.device, dtype=cross_dtype)
        image_features = self.model.cross_model(image_f, text_f, mask)[:, 0, :].float()
        text_features = text_features.to(image_features.device).float()
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        similarity = (text_features * image_features).sum(dim=-1)
        return self.model.logit_scale.exp().float() * similarity
