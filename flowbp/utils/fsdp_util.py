# ruff: noqa: E731


import functools
from functools import partial

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock

try:
    from diffusers.models.attention import JointTransformerBlock
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    _SD3_AVAILABLE = True
except ImportError:
    JointTransformerBlock = None
    SD3Transformer2DModel = None
    _SD3_AVAILABLE = False

try:
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2SingleTransformerBlock,
        Flux2Transformer2DModel,
        Flux2TransformerBlock,
    )

    _FLUX2_AVAILABLE = True
except ImportError:
    Flux2Transformer2DModel = None
    Flux2TransformerBlock = None
    Flux2SingleTransformerBlock = None
    _FLUX2_AVAILABLE = False

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

def _declared_no_split_modules(transformer):
    """Resolve ``model._no_split_modules`` class names to the actual classes.

    Diffusers transformers declare their FSDP-wrappable block types by name.
    Resolving them off the instantiated model keeps new backbones working
    without another hardcoded branch here.
    """
    names = list(getattr(transformer, "_no_split_modules", None) or [])
    if not names:
        return ()
    found = {}
    for module in transformer.modules():
        cls = type(module)
        if cls.__name__ in names and cls.__name__ not in found:
            found[cls.__name__] = cls
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(
            f"{type(transformer).__name__} declares _no_split_modules={names}, "
            f"but these classes were not found in the instantiated model: {missing}"
        )
    return tuple(found[name] for name in names)


def get_no_split_modules(transformer):
    peft_base_model = getattr(transformer, "base_model", None)
    if peft_base_model is not None:
        transformer = getattr(peft_base_model, "model", transformer)
    if isinstance(transformer, FluxTransformer2DModel):
        return (FluxTransformerBlock, FluxSingleTransformerBlock)
    if _FLUX2_AVAILABLE and isinstance(transformer, Flux2Transformer2DModel):
        return (Flux2TransformerBlock, Flux2SingleTransformerBlock)
    if _SD3_AVAILABLE and isinstance(transformer, SD3Transformer2DModel):
        return (JointTransformerBlock,)

    declared = _declared_no_split_modules(transformer)
    if declared:
        return declared
    raise ValueError(
        f"Unsupported transformer type: {type(transformer)}. It declares no "
        "_no_split_modules, so FSDP has no block boundary to shard on."
    )


def apply_fsdp_checkpointing(model, no_split_modules, p=1):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print("--> applying fdsp activation checkpointing...")
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        # Gradient communication precision.
        reduce_dtype=torch.float32,
        # Buffer precision.
        buffer_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    return mixed_precision


def get_dit_fsdp_kwargs(
    transformer,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
    ignored_states=None,
):
    no_split_modules = get_no_split_modules(transformer)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=no_split_modules,
    )

    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    device_id = torch.cuda.current_device()
    cpu_offload = (torch.distributed.fsdp.CPUOffload(
        offload_params=True) if cpu_offload else None)
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
    }

    # PEFT adapters mix frozen base parameters with trainable adapter parameters
    # inside each DiT block, so preserve original parameters under FSDP.
    if use_lora:
        fsdp_kwargs.update({
            "use_orig_params": True,
            "sync_module_states": True,
        })
    if ignored_states:
        fsdp_kwargs["ignored_states"] = list(ignored_states)

    return fsdp_kwargs, no_split_modules


def get_discriminator_fsdp_kwargs(master_weight_type="fp32"):
    auto_wrap_policy = None

    # Use existing mixed precision settings

    mixed_precision = get_mixed_precision(master_weight_type)
    sharding_strategy = ShardingStrategy.NO_SHARD
    device_id = torch.cuda.current_device()
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
    }

    return fsdp_kwargs
    
