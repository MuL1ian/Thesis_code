"""Compatibility classes required to unpickle the public MPS checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import sys
import types
from typing import Optional

import torch
import torch.nn.functional as F
from torch import einsum, nn
from transformers import CLIPConfig
from transformers import CLIPModel as HFCLIPModel


def normalize_hf_clip_configs(loaded_object, default_attn_implementation: str = "eager"):
    """Patch configs serialized by older transformers releases."""

    seen_configs: set[int] = set()

    def patch_config(config) -> None:
        if config is None or id(config) in seen_configs:
            return
        seen_configs.add(id(config))
        data = getattr(config, "__dict__", None)
        if not isinstance(data, dict):
            return
        if "_attn_implementation_internal" not in data:
            setattr(
                config,
                "_attn_implementation_internal",
                data.get("attn_implementation")
                or data.get("_attn_implementation")
                or default_attn_implementation,
            )
        for child_name in ("text_config", "vision_config"):
            patch_config(data.get(child_name))

    def patch_module(module: nn.Module) -> None:
        for submodule in module.modules():
            patch_config(getattr(submodule, "config", None))
            if submodule.__class__.__name__ == "CLIPAttention" and not hasattr(
                submodule, "is_causal",
            ):
                submodule.is_causal = False
            if submodule.__class__.__name__ == "CLIPTextTransformer" and not hasattr(
                submodule, "eos_token_id",
            ):
                config = getattr(submodule, "config", None)
                submodule.eos_token_id = int(getattr(config, "eos_token_id", 2))

    def walk(value) -> None:
        if isinstance(value, nn.Module):
            patch_module(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)
        else:
            patch_config(getattr(value, "config", None))

    walk(loaded_object)
    return loaded_object


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("bias", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len: int, *, device: torch.device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i, j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], 2, x.shape[-1] // 2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, tensor):
    return (tensor * pos.cos()) + (rotate_half(tensor) * pos.sin())


def _split_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    batch, tokens, hidden = tensor.shape
    dim_head = hidden // heads
    return tensor.view(batch, tokens, heads, dim_head).permute(0, 2, 1, 3)


def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, dim_head = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim_head)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.norm = LayerNorm(dim)
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, ff_inner_dim * 2)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False),
        )
        self.register_buffer("pos_emb", None, persistent=False)

    def get_rotary_embedding(self, n: int, device: torch.device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]
        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, attn_mask=None):
        n, device, heads = x.shape[1], x.device, self.heads
        x = self.norm(x)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
        q = _split_heads(q, heads=heads)
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda tensor: apply_rotary_pos_emb(positions, tensor), (q, k))
        sim = einsum("b h i d, b j d -> b h i j", q * self.scale, k)
        if attn_mask is not None:
            sim = sim.masked_fill(
                ~attn_mask.unsqueeze(1),
                -torch.finfo(sim.dtype).max,
            )
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b j d -> b h i d", attn, v)
        return self.attn_out(_merge_heads(out)) + self.ff_out(ff)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        context_dim: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 12,
        parallel_ff: bool = False,
        ff_mult: int = 4,
        norm_context: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else dim
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        ff_inner_dim = ff_mult * dim
        self.ff = (
            nn.Sequential(
                nn.Linear(dim, ff_inner_dim * 2, bias=False),
                SwiGLU(),
                nn.Linear(ff_inner_dim, dim, bias=False),
            )
            if parallel_ff
            else None
        )

    def forward(self, x, context, mask):
        x = self.norm(x)
        context = self.context_norm(context)
        q = _split_heads(self.to_q(x), heads=self.heads) * self.scale
        k, v = self.to_kv(context).chunk(2, dim=-1)
        sim = einsum("b h i d, b j d -> b h i j", q, k)
        sim = sim + mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)
        out = self.to_out(_merge_heads(einsum("b h i j, b j d -> b h i d", attn, v)))
        if self.ff is not None:
            out = out + self.ff(x)
        return out


class Cross_model(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        layer_num: int = 4,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(
                            CrossAttention(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                parallel_ff=True,
                                ff_mult=ff_mult,
                            )
                        ),
                        Residual(
                            ParallelTransformerBlock(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                ff_mult=ff_mult,
                            )
                        ),
                    ]
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, query_tokens, context_tokens, mask):
        for cross_attn, self_attn_ff in self.layers:
            query_tokens = cross_attn(query_tokens, context_tokens, mask)
            query_tokens = self_attn_ff(query_tokens)
        return query_tokens


class XCLIPModel(HFCLIPModel):
    @staticmethod
    def _config_value(config, public_name: str, private_name: str, default: bool):
        data = getattr(config, "__dict__", {})
        if private_name in data:
            return bool(data[private_name])
        if public_name in data:
            return bool(data[public_name])
        try:
            return bool(getattr(config, public_name))
        except AttributeError:
            return default

    @staticmethod
    def _call_with_supported_kwargs(module: nn.Module, **kwargs):
        try:
            params = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            params = {}
        return module(**{key: value for key, value in kwargs.items() if key in params})

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self._config_value(self.config, "output_attentions", "_output_attentions", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self._config_value(
                self.config,
                "output_hidden_states",
                "_output_hidden_states",
                False,
            )
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self._config_value(self.config, "use_return_dict", "return_dict", False)
        )
        outputs = self._call_with_supported_kwargs(
            self.text_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return self.text_projection(outputs[0]), self.text_projection(outputs[1])

    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self._config_value(self.config, "output_attentions", "_output_attentions", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self._config_value(
                self.config,
                "output_hidden_states",
                "_output_hidden_states",
                False,
            )
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self._config_value(self.config, "use_return_dict", "return_dict", False)
        )
        outputs = self._call_with_supported_kwargs(
            self.vision_model,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return self.visual_projection(outputs[0])


@dataclass
class BaseModelConfig:
    pass


@dataclass
class ClipModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.clip_model.CLIPModel"
    pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32"


class CLIPModel(nn.Module):
    def __init__(self, ckpt: str):
        super().__init__()
        self.model = XCLIPModel.from_pretrained(ckpt)
        self.cross_model = Cross_model(dim=1024, layer_num=4, heads=16)

    @property
    def logit_scale(self):
        return self.model.logit_scale


def ensure_mps_compat_modules() -> None:
    """Register the module names embedded in the official pickle."""

    this_module = sys.modules[__name__]
    trainer_module = sys.modules.get("trainer")
    if trainer_module is None:
        trainer_module = types.ModuleType("trainer")
        trainer_module.__path__ = []
        sys.modules["trainer"] = trainer_module

    models_module = sys.modules.get("trainer.models")
    if models_module is None:
        models_module = types.ModuleType("trainer.models")
        models_module.__path__ = []
        sys.modules["trainer.models"] = models_module
    trainer_module.models = models_module

    for module_name in (
        "trainer.models.base_model",
        "trainer.models.cross_modeling",
        "trainer.models.clip_model",
    ):
        sys.modules.setdefault(module_name, this_module)
    models_module.base_model = sys.modules["trainer.models.base_model"]
    models_module.cross_modeling = sys.modules["trainer.models.cross_modeling"]
    models_module.clip_model = sys.modules["trainer.models.clip_model"]
