"""HPSv3++ reward scorer.

This module is intentionally separate from ``hpsv3_scorer.py``.  The installed
``hpsv3`` package in our runtime supports the original HPSv3/Qwen2-VL path, while
HPSv3++ uses a Qwen3-VL backbone plus capability and FiLM conditioning heads.
The implementation below mirrors the public HPSv3++ inference code but keeps it
local to FlowBP so SD3.5 training can use the reward as a differentiable module.
"""

from __future__ import annotations

import os
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import get_kbit_device_map, get_quantization_config

from flowbp.eval.rewards.reward_ckpt_path import CKPT_PATH


def _patch_transformers_for_hpsv3plusplus() -> None:
    """Backfill typing aliases before any ``hpsv3`` import.

    The installed ``hpsv3`` package imports
    ``transformers.image_utils.VideoInput`` at module load time. Some
    transformers builds used by FlowBP omit that alias, so patch it first.
    """

    try:
        from transformers import image_utils as hf_image_utils
    except Exception:
        return
    if not hasattr(hf_image_utils, "VideoInput"):
        hf_image_utils.VideoInput = object


# Apply early so ``import flowbp.eval.rewards.hpsv3plusplus_scorer`` is enough
# to make subsequent ``hpsv3`` imports succeed in the flowdmd environment.
_patch_transformers_for_hpsv3plusplus()


class Qwen3VLRewardModelBT(Qwen3VLForConditionalGeneration):
    """Qwen3-VL reward model with a pooled reward head."""

    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
        **kwargs,
    ):
        del kwargs
        super().__init__(config)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = config.text_config.hidden_size

        self.output_dim = output_dim
        if rm_head_type == "default":
            self.rm_head = nn.Linear(hidden_size, output_dim, bias=False)
        elif rm_head_type == "ranknet":
            if rm_head_kwargs is not None:
                layers: list[nn.Module] = []
                in_dim = hidden_size
                num_layers = int(rm_head_kwargs.get("num_layers", 3))
                for layer_idx in range(num_layers):
                    if layer_idx < num_layers - 1:
                        out_dim = int(rm_head_kwargs["hidden_size"])
                        layers.extend(
                            [
                                nn.Linear(in_dim, out_dim),
                                nn.ReLU(),
                                nn.Dropout(float(rm_head_kwargs.get("dropout", 0.1))),
                            ]
                        )
                        in_dim = out_dim
                    else:
                        layers.append(
                            nn.Linear(
                                in_dim,
                                output_dim,
                                bias=bool(rm_head_kwargs.get("bias", False)),
                            )
                        )
                self.rm_head = nn.Sequential(*layers)
            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )
        else:
            raise ValueError(f"Invalid rm_head_type: {rm_head_type}")

        self.rm_head.to(torch.float32)
        self.reward_token = "special" if special_token_ids is not None else reward_token
        self.special_token_ids = special_token_ids

    def _pool_hidden(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None):
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("HPSv3++ reward pooling requires input_ids.")

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (
                torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            )
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(hidden_states.device)

        if self.reward_token == "last":
            return hidden_states[
                torch.arange(batch_size, device=hidden_states.device), sequence_lengths
            ]
        if self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=hidden_states.size(1) - 1)
            return torch.stack(
                [hidden_states[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)]
            )
        if self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_hidden = hidden_states[special_token_mask, ...]
            return pooled_hidden.view(batch_size, -1)
        raise ValueError(f"Invalid reward_token: {self.reward_token}")

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cond_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del return_dict, cond_values, kwargs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        pooled_hidden = self._pool_hidden(outputs[0], input_ids)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = self.rm_head(pooled_hidden)
        return {"logits": logits}


class Qwen3VLRewardModelFiLMContinuous(Qwen3VLRewardModelBT):
    """Qwen3-VL reward model with FiLM conditioning."""

    def __init__(self, *args, cond_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.config.text_config.hidden_size
        self.cond_encoder = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

    def _condition_hidden(
        self,
        pooled_hidden: torch.Tensor,
        cond_values: Optional[torch.Tensor],
        iter_values: Optional[torch.Tensor],
        level_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        batch_size = pooled_hidden.shape[0]
        if cond_values is None:
            if level_ids is not None:
                level_to_capability = {0: 0.0, 1: 0.5, 2: 1.0}
                caps = torch.tensor(
                    [level_to_capability.get(int(level.item()), 0.5) for level in level_ids],
                    dtype=torch.float32,
                    device=pooled_hidden.device,
                )
                iters = (
                    iter_values.to(device=pooled_hidden.device, dtype=torch.float32)
                    if iter_values is not None
                    else torch.zeros_like(caps)
                )
                cond_values = torch.stack([caps, iters], dim=-1)
            elif iter_values is not None:
                iters = iter_values.to(device=pooled_hidden.device, dtype=torch.float32)
                caps = torch.full_like(iters, 0.5, dtype=torch.float32)
                cond_values = torch.stack([caps, iters], dim=-1)
            else:
                cond_values = torch.tensor(
                    [[0.5, 0.0]] * batch_size,
                    dtype=torch.float32,
                    device=pooled_hidden.device,
                )

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            cond_emb = self.cond_encoder(cond_values.float())
            film_params = self.film_gen(cond_emb)
            gamma, beta = film_params.chunk(2, dim=-1)
            return pooled_hidden.float() * (1.0 + gamma) + beta

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cond_values: Optional[torch.Tensor] = None,
        iter_values: Optional[torch.Tensor] = None,
        level_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        del return_dict, kwargs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        pooled_hidden = self._pool_hidden(outputs[0], input_ids)
        conditioned = self._condition_hidden(pooled_hidden, cond_values, iter_values, level_ids)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = self.rm_head(conditioned)
        return {"logits": logits}


class CapabilityEncoder(nn.Module):
    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, cond_dim)
        self.pool_attn = nn.Linear(cond_dim, 1)
        self.out_proj = nn.Linear(cond_dim, 1)

    def forward(self, group_features: torch.Tensor) -> torch.Tensor:
        x = self.proj(group_features.to(dtype=self.proj.weight.dtype))
        attn = torch.softmax(self.pool_attn(x), dim=0)
        group_emb = (x * attn).sum(dim=0)
        return torch.sigmoid(self.out_proj(group_emb)).squeeze(-1).to(dtype=torch.float32)


class Qwen3VLRewardModelFiLMHybrid(Qwen3VLRewardModelFiLMContinuous):
    """HPSv3++ film_hybrid model: inferred capability plus explicit RL iter."""

    def __init__(self, *args, cond_dim=256, **kwargs):
        super().__init__(*args, cond_dim=cond_dim, **kwargs)
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.config.text_config.hidden_size
        self.cap_encoder = CapabilityEncoder(hidden_size, cond_dim)
        self.cap_encoder.to(torch.float32)

    def _infer_capability(self, pooled_hidden: torch.Tensor, group_ids=None) -> torch.Tensor:
        pooled_detached = pooled_hidden.detach()
        batch_size = pooled_detached.shape[0]
        device = pooled_detached.device
        caps = torch.zeros(batch_size, dtype=torch.float32, device=device)
        if getattr(self, "per_image_cap", False) or (group_ids is None and batch_size == 1):
            for i in range(batch_size):
                caps[i] = self.cap_encoder(pooled_detached[i : i + 1])
        elif group_ids is not None:
            group_ids = group_ids.to(device).long()
            for group_id in torch.unique(group_ids, sorted=True):
                mask = group_ids == group_id
                caps[mask] = self.cap_encoder(pooled_detached[mask])
        else:
            caps[:] = self.cap_encoder(pooled_detached)
        return caps

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cond_values: Optional[torch.Tensor] = None,
        iter_values: Optional[torch.Tensor] = None,
        level_ids: Optional[torch.LongTensor] = None,
        group_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        del return_dict, cond_values, level_ids, kwargs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        pooled_hidden = self._pool_hidden(outputs[0], input_ids)
        cap_pred = self._infer_capability(pooled_hidden, group_ids=group_ids)
        if iter_values is None:
            iter_val = torch.zeros(pooled_hidden.shape[0], dtype=torch.float32, device=pooled_hidden.device)
        else:
            iter_val = iter_values.to(device=pooled_hidden.device, dtype=torch.float32)
        cond_values = torch.stack([cap_pred, iter_val], dim=-1)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            conditioned = self._condition_hidden(pooled_hidden, cond_values, None, None)
            logits = self.rm_head(conditioned)
        return {"logits": logits, "cond_pred": cond_values}


def _first_existing_file(*paths: str | os.PathLike | None) -> str:
    for path in paths:
        if path and Path(path).expanduser().is_file():
            return str(Path(path).expanduser())
    return ""


def discover_hpsv3plusplus_paths(
    reward_ckpt_path: str = "",
    config_path: str = "",
    checkpoint_path: str = "",
) -> tuple[str, str]:
    root = Path(reward_ckpt_path or CKPT_PATH).expanduser()
    cfg = _first_existing_file(
        os.environ.get("FLOWBP_HPSV3PLUSPLUS_CONFIG_PATH"),
        os.environ.get("FLOWBP_HPSV3_CONFIG_PATH"),
        config_path,
        root / "HPSv3-PlusPlus" / "config.json",
        root / "hpsv3plusplus" / "config.json",
    )
    ckpt = _first_existing_file(
        os.environ.get("FLOWBP_HPSV3PLUSPLUS_CHECKPOINT_PATH"),
        os.environ.get("FLOWBP_HPSV3_CHECKPOINT_PATH"),
        checkpoint_path,
        root / "HPSv3-PlusPlus" / "hpsv3++.pth",
        root / "hpsv3plusplus" / "hpsv3++.pth",
    )
    return cfg, ckpt


def _load_state_dict(path: str):
    if path.endswith(".safetensors"):
        import safetensors.torch

        state_dict = safetensors.torch.load_file(path, device="cpu")
    else:
        state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def _prepare_input(data, device: str):
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    if isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    if isinstance(data, torch.Tensor):
        return data.to(device=device)
    return data


class HPSv3PlusPlusInferencer:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        differentiable: bool = False,
        iter_step: float = 0.0,
    ):
        _patch_transformers_for_hpsv3plusplus()
        from hpsv3.dataset.data_collator_qwen import (
            INSTRUCTION,
            prompt_with_special_token,
            prompt_without_special_token,
        )
        from hpsv3.dataset.utils import process_vision_info
        from hpsv3.model.differentiable_image_processor import Qwen2VLImageProcessor
        from hpsv3.utils.parser import (
            DataConfig,
            ModelConfig,
            PEFTLoraConfig,
            TrainingConfig,
            parse_args_with_yaml,
        )

        self._instruction = INSTRUCTION
        self._prompt_with_special_token = prompt_with_special_token
        self._prompt_without_special_token = prompt_without_special_token
        self._process_vision_info = process_vision_info
        self.device = str(device)
        self.default_iter_step = float(iter_step)

        (data_config, training_args, model_config, peft_lora_config), _ = parse_args_with_yaml(
            (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig),
            config_path,
            is_train=False,
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        model_type = str(raw_config.get("model_type", ""))
        cond_dim = int(raw_config.get("cond_dim", 256))

        torch_dtype = (
            model_config.torch_dtype
            if model_config.torch_dtype in ("auto", None)
            else getattr(torch, model_config.torch_dtype)
        )
        quantization_config = get_quantization_config(model_config)
        processor = AutoProcessor.from_pretrained(
            model_config.model_name_or_path,
            padding_side="right",
            trust_remote_code=True,
        )
        if differentiable:
            original_image_processor = processor.image_processor
            min_pixels = getattr(original_image_processor, "min_pixels", None)
            if min_pixels is None:
                min_pixels = int(raw_config.get("min_pixels", 56 * 56))
            max_pixels = getattr(original_image_processor, "max_pixels", None)
            if max_pixels is None:
                max_pixels = int(raw_config.get("max_pixels", 28 * 28 * 1280))
            processor.image_processor = Qwen2VLImageProcessor(
                do_resize=getattr(original_image_processor, "do_resize", True),
                # Differentiable training passes decoded tensors already in [0, 1].
                # Do not apply the PIL-style 1/255 rescale again.
                do_rescale=False,
                rescale_factor=getattr(original_image_processor, "rescale_factor", 1 / 255),
                do_normalize=getattr(original_image_processor, "do_normalize", True),
                image_mean=getattr(original_image_processor, "image_mean", None),
                image_std=getattr(original_image_processor, "image_std", None),
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                patch_size=getattr(original_image_processor, "patch_size", 16),
                temporal_patch_size=getattr(original_image_processor, "temporal_patch_size", 2),
                merge_size=getattr(
                    original_image_processor,
                    "merge_size",
                    getattr(original_image_processor, "spatial_merge_size", 2),
                ),
            )

        special_token_ids = None
        if model_config.use_special_tokens:
            special_tokens = ["<|Reward|>"]
            processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )
            special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

        model_cls = Qwen3VLRewardModelBT
        if model_type == "film_hybrid":
            model_cls = Qwen3VLRewardModelFiLMHybrid
        elif model_type in {"film", "film_continuous"}:
            model_cls = Qwen3VLRewardModelFiLMContinuous

        model_kwargs = dict(
            revision=getattr(model_config, "model_revision", None),
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
            use_cache=False,
        )
        model = model_cls.from_pretrained(
            model_config.model_name_or_path,
            output_dim=model_config.output_dim,
            reward_token=model_config.reward_token,
            special_token_ids=special_token_ids,
            torch_dtype=torch_dtype,
            attn_implementation=(
                "flash_attention_2"
                if not training_args.disable_flash_attn2
                else "sdpa"
            ),
            rm_head_type=model_config.rm_head_type,
            rm_head_kwargs=model_config.rm_head_kwargs,
            cond_dim=cond_dim,
            trust_remote_code=True,
            **model_kwargs,
        )
        if model_config.use_special_tokens:
            model.resize_token_embeddings(len(processor.tokenizer))
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
        model.rm_head.to(torch.float32)
        if hasattr(model, "cap_encoder"):
            model.cap_encoder.to(torch.float32)
        if hasattr(model, "cond_encoder"):
            model.cond_encoder.to(torch.float32)
        if hasattr(model, "film_gen"):
            model.film_gen.to(torch.float32)
        model.config.tokenizer_padding_side = processor.tokenizer.padding_side
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        state_dict = _load_state_dict(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self.device)
        self.model = model
        self.processor = processor
        self.data_config = data_config
        self.use_special_tokens = model_config.use_special_tokens

    def _prepare_inputs(self, inputs):
        inputs = _prepare_input(inputs, self.device)
        if len(inputs) == 0:
            raise ValueError("Empty HPSv3++ input batch.")
        return inputs

    def prepare_batch(self, images, prompts: Sequence[str]):
        tensor_inputs = all(isinstance(image, torch.Tensor) for image in images)
        image_processor = getattr(self.processor, "image_processor", None)
        old_do_rescale = getattr(image_processor, "do_rescale", None)
        # Differentiable training sends decoded tensors already in [0, 1], while
        # online eval sends PIL images in [0, 255]. Use the same inferencer for
        # both without double-rescaling tensors or under-rescaling PIL images.
        if old_do_rescale is not None:
            image_processor.do_rescale = not tensor_inputs
        max_pixels = 256 * 28 * 28
        messages = []
        try:
            for prompt, image in zip(prompts, images):
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                    "min_pixels": max_pixels,
                                    "max_pixels": max_pixels,
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        self._instruction.format(text_prompt=prompt)
                                        + self._prompt_with_special_token
                                        if self.use_special_tokens
                                        else self._prompt_without_special_token
                                    ),
                                },
                            ],
                        }
                    ]
                )
            image_inputs, _ = self._process_vision_info(messages)
            batch = self.processor(
                text=self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                videos_kwargs={"do_rescale": True},
            )
            return self._prepare_inputs(batch)
        finally:
            if old_do_rescale is not None:
                image_processor.do_rescale = old_do_rescale

    def reward(self, images, prompts: Sequence[str], iter_step=None):
        batch = self.prepare_batch(images, prompts)
        batch_size = batch["input_ids"].shape[0]
        if iter_step is None:
            iter_step = self.default_iter_step
        if not torch.is_tensor(iter_step):
            iter_values = torch.full((batch_size,), float(iter_step), dtype=torch.float32)
        else:
            iter_values = iter_step.float().reshape(-1)
        iter_values = iter_values.to(self.device)
        if iter_values.numel() == 1 and batch_size > 1:
            iter_values = iter_values.expand(batch_size)
        return self.model(return_dict=True, iter_values=iter_values, **batch)["logits"]


class HPSv3PlusPlusScorer(torch.nn.Module):
    def __init__(
        self,
        dtype,
        device,
        reward_ckpt_path: str = "",
        config_path: str = "",
        checkpoint_path: str = "",
        differentiable: bool = False,
        iter_step: float = 0.0,
    ):
        super().__init__()
        del dtype
        cfg, ckpt = discover_hpsv3plusplus_paths(
            reward_ckpt_path=reward_ckpt_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )
        if not cfg or not ckpt:
            raise FileNotFoundError(
                "Could not find HPSv3++ config/checkpoint. Expected "
                "<reward_ckpt_path>/HPSv3-PlusPlus/{config.json,hpsv3++.pth} "
                "or explicit config/checkpoint paths."
            )
        self.device = torch.device(device)
        self._inferencer = HPSv3PlusPlusInferencer(
            config_path=cfg,
            checkpoint_path=ckpt,
            device=str(self.device),
            differentiable=differentiable,
            iter_step=iter_step,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                device = first
        if device is not None:
            self.device = torch.device(device)
            self._inferencer.device = str(self.device)
            self._inferencer.model.to(self._inferencer.device)
        return self

    def __call__(self, images, prompts, iter_step=None):
        rewards = self._inferencer.reward(images, prompts, iter_step=iter_step)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, device=self.device)
        rewards = rewards.float()
        if rewards.ndim == 2 and rewards.shape[1] >= 1:
            rewards = rewards[:, 0]
        elif rewards.ndim > 1:
            rewards = rewards.reshape(rewards.shape[0], -1)[:, 0]
        return rewards.contiguous()
