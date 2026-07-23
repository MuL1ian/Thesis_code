from __future__ import annotations

from typing import Iterable

from torch import nn

from flowbp.utils.logging_ import main_print


ATTENTION_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_out.0",
    "to_add_out",
)

DEFAULT_TARGET_MODULES = (
    *ATTENTION_TARGET_MODULES,
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
)

_TARGET_MODULE_PRESETS = {
    "default": DEFAULT_TARGET_MODULES,
    "attention": ATTENTION_TARGET_MODULES,
}


def _split_target_modules(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_TARGET_MODULES)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return list(DEFAULT_TARGET_MODULES)
        preset = _TARGET_MODULE_PRESETS.get(value.lower())
        if preset is not None:
            return list(preset)
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _validate_target_modules(transformer, target_modules: list[str]) -> list[str]:
    matched_targets: set[str] = set()
    matched_linear_names: list[str] = []
    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for target in target_modules:
            if name == target or name.endswith(f".{target}"):
                matched_targets.add(target)
                matched_linear_names.append(name)

    if not matched_linear_names:
        raise ValueError(
            "Adapter matched no nn.Linear modules. Check target_modules "
            f"against {type(transformer).__name__}."
        )

    unmatched = [target for target in target_modules if target not in matched_targets]
    if unmatched:
        main_print(f"[warning] Adapter target suffixes matched no Linear modules: {unmatched}")
    main_print(
        f"--> matched {len(matched_linear_names)} Linear modules "
        f"using {len(matched_targets)} target suffixes"
    )
    return [target for target in target_modules if target in matched_targets]


def apply_lora_if_enabled(transformer, args):
    """Optionally wrap a diffusion transformer with a standard LoRA adapter."""
    if not bool(getattr(args, "use_lora", False)):
        return transformer, False

    from peft import LoraConfig, get_peft_model

    target_modules = _split_target_modules(getattr(args, "lora_target_modules", None))
    target_modules = _validate_target_modules(transformer, target_modules)
    base_config = getattr(transformer, "config", None)
    main_print(
        "--> applying LoRA adapter "
        f"(rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, targets={target_modules})"
    )
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=None,
        ),
    )

    if base_config is not None:
        try:
            transformer.config = base_config
        except Exception:
            pass
    transformer.print_trainable_parameters()
    return transformer, True


def apply_oft_if_enabled(transformer, args):
    """Optionally wrap a diffusion transformer with a standard OFT adapter."""
    if not bool(getattr(args, "use_oft", False)):
        return transformer, False

    from peft import OFTConfig, get_peft_model

    target_modules = _split_target_modules(getattr(args, "oft_target_modules", None))
    target_modules = _validate_target_modules(transformer, target_modules)
    base_config = getattr(transformer, "config", None)
    main_print(
        "--> applying OFT adapter "
        f"(block_size={args.oft_block_size}, dropout={args.oft_module_dropout}, "
        f"coft={args.oft_coft}, targets={target_modules})"
    )
    transformer = get_peft_model(
        transformer,
        OFTConfig(
            r=0,
            oft_block_size=args.oft_block_size,
            module_dropout=args.oft_module_dropout,
            target_modules=target_modules,
            bias="none",
            coft=args.oft_coft,
            eps=args.oft_eps,
            block_share=args.oft_block_share,
            task_type=None,
        ),
    )

    if base_config is not None:
        try:
            transformer.config = base_config
        except Exception:
            pass
    transformer.print_trainable_parameters()
    return transformer, True
