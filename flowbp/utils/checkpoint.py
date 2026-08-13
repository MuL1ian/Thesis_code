import copy
from contextlib import contextmanager
import json
import math
import os
import re
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.fsdp import (FullOptimStateDictConfig,
                                    FullStateDictConfig)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from flowbp.utils.logging_ import main_print

WEIGHT_FILENAME = "diffusion_pytorch_model.safetensors"
TRAINING_STATE_FILENAME = "training_state.pt"
OPTIMIZER_FILENAME = "optimizer.pt"
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)-(\d+)$")


@contextmanager
def atomic_output_dir(final_dir):
    """Build a checkpoint beside its final path, then publish it atomically."""
    final_dir = os.path.abspath(os.fspath(final_dir))
    parent = os.path.dirname(final_dir)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(final_dir):
        raise FileExistsError(f"Refusing to overwrite checkpoint: {final_dir}")

    temp_dir = tempfile.mkdtemp(
        prefix=f".{os.path.basename(final_dir)}.tmp-",
        dir=parent,
    )
    try:
        yield temp_dir
        os.rename(temp_dir, final_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def sync_rank0_save_error(rank, error_message, context):
    """Make every rank fail together when rank 0 cannot publish a checkpoint."""
    payload = [error_message if rank <= 0 else None]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"{context} failed on rank 0: {payload[0]}")


def _state_key_ending_with(state_dict, suffix):
    matches = [key for key in state_dict if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one state-dict key ending with {suffix!r}, got {matches}"
        )
    return matches[0]


def _canonical_module_name(module_name):
    """Remove wrapper-only path segments omitted by FSDP full state dicts."""
    wrapper_segments = {
        "_fsdp_wrapped_module",
        "_checkpoint_wrapped_module",
    }
    return ".".join(
        segment
        for segment in str(module_name).split(".")
        if segment not in wrapper_segments
    )


def _save_portable_loraga_checkpoint(
    peft_model,
    full_state_dict,
    save_dir,
    step,
    epoch,
):
    """Save LoRA-GA relative to the original base as standard rank-2r LoRA."""
    from flowbp.utils.gradlora_adapter import (
        _active_adapter_name,
        _is_lora_linear,
    )
    from flowbp.utils.loraga_adapter import (
        LORAGA_INIT_A_ATTR,
        LORAGA_INIT_B_ATTR,
        convert_loraga_to_portable_factors,
    )

    portable_state = dict(full_state_dict)
    init_state = {}
    adapters = set()
    original_rank = None
    for module_name, module in peft_model.named_modules():
        if not _is_lora_linear(module):
            continue
        adapter = _active_adapter_name(module)
        adapters.add(adapter)
        state_module_name = _canonical_module_name(module_name)
        a_suffix = f"{state_module_name}.lora_A.{adapter}.weight"
        b_suffix = f"{state_module_name}.lora_B.{adapter}.weight"
        a_key = _state_key_ending_with(portable_state, a_suffix)
        b_key = _state_key_ending_with(portable_state, b_suffix)
        init_a = getattr(module, LORAGA_INIT_A_ATTR, None)
        init_b = getattr(module, LORAGA_INIT_B_ATTR, None)
        if not isinstance(init_a, torch.Tensor) or not isinstance(
            init_b, torch.Tensor
        ):
            raise RuntimeError(
                f"LoRA-GA layer {module_name} is missing initial factors"
            )
        final_a = portable_state[a_key].detach().cpu()
        final_b = portable_state[b_key].detach().cpu()
        portable_a, portable_b = convert_loraga_to_portable_factors(
            final_a,
            final_b,
            init_a,
            init_b,
        )
        portable_state[a_key] = portable_a
        portable_state[b_key] = portable_b
        init_state[a_key] = init_a
        init_state[b_key] = init_b
        layer_rank = int(final_a.shape[0])
        if original_rank is None:
            original_rank = layer_rank
        elif original_rank != layer_rank:
            raise RuntimeError("Portable LoRA-GA requires a uniform adapter rank")

    if not adapters or original_rank is None:
        raise RuntimeError("Portable LoRA-GA save found no initialized layers")
    if len(adapters) != 1:
        raise RuntimeError(
            f"Portable LoRA-GA currently supports one adapter, got {adapters}"
        )
    adapter = next(iter(adapters))
    config = peft_model.peft_config[adapter]
    original_config = copy.deepcopy(config)
    use_rslora = bool(getattr(config, "use_rslora", False))
    alpha = float(config.lora_alpha)
    portable_alpha = alpha * (math.sqrt(2.0) if use_rslora else 2.0)
    try:
        config.r = 2 * original_rank
        config.lora_alpha = portable_alpha
        rank_pattern = getattr(config, "rank_pattern", None)
        if isinstance(rank_pattern, dict):
            config.rank_pattern = {
                key: 2 * int(value) for key, value in rank_pattern.items()
            }
        alpha_pattern = getattr(config, "alpha_pattern", None)
        if isinstance(alpha_pattern, dict):
            factor = math.sqrt(2.0) if use_rslora else 2.0
            config.alpha_pattern = {
                key: float(value) * factor
                for key, value in alpha_pattern.items()
            }
        peft_model.save_pretrained(
            save_dir,
            state_dict=portable_state,
            safe_serialization=True,
            is_main_process=True,
        )
    finally:
        peft_model.peft_config[adapter] = original_config

    save_file(
        {key: value.contiguous() for key, value in init_state.items()},
        os.path.join(save_dir, "loraga_initial_factors.safetensors"),
    )
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump(
            {
                "step": step,
                "epoch": epoch,
                "adapter_type": "loraga",
                "training_rank": original_rank,
                "portable_rank": 2 * original_rank,
                "training_alpha": alpha,
                "portable_alpha": portable_alpha,
                "use_rslora": use_rslora,
            },
            f,
            indent=4,
        )


def _unwrap_model_config(model):
    for attr in ("module",):
        model = getattr(model, attr, model)

    config = getattr(model, "config", None)
    if config is not None:
        return config

    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        inner_model = getattr(base_model, "model", base_model)
        config = getattr(inner_model, "config", None)
        if config is not None:
            return config

    raise AttributeError(f"Could not find a diffusers config on {type(model)}")


def _config_to_dict(model):
    config_dict = dict(_unwrap_model_config(model))
    config_dict.pop("dtype", None)
    return config_dict


def save_checkpoint_optimizer(model,
                              optimizer,
                              rank,
                              output_dir,
                              step,
                              discriminator=False):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(
            model,
            optimizer,
        )

    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    if rank <= 0 and not discriminator:
        weight_path = os.path.join(save_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = _config_to_dict(model)
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        optimizer_path = os.path.join(save_dir, "optimizer.pt")
        torch.save(optim_state, optimizer_path)
    else:
        weight_path = os.path.join(save_dir,
                                   "discriminator_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        optimizer_path = os.path.join(save_dir, "discriminator_optimizer.pt")
        torch.save(optim_state, optimizer_path)
    main_print(f"--> checkpoint saved at step {step}")


def save_checkpoint(transformer,
                    rank,
                    output_dir,
                    step,
                    epoch,
                    mean_grad_norm=None,
                    optimizer=None,
                    lr_scheduler=None,
                    save_optimizer_state=False):
    """Save transformer weights, plus the state needed by `--resume_from_checkpoint`.

    ``optimizer`` is only serialized when ``save_optimizer_state`` is set,
    because a full AdamW state is roughly twice the parameter footprint.
    Without it a resumed run restores weights, step counter and LR schedule but
    restarts the AdamW moments from zero. Data-loader and RNG state are not
    checkpointed, so even optimizer-state resume is not bitwise reproducible.
    """
    main_print(f"--> saving checkpoint at step {step}")
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = transformer.state_dict()

    # FSDP.optim_state_dict is a collective, so every rank has to enter it even
    # though only rank 0 ends up holding the gathered state.
    optim_state = None
    if save_optimizer_state and optimizer is not None:
        with FSDP.state_dict_type(
                transformer,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            optim_state = FSDP.optim_state_dict(transformer, optimizer)

    save_error = None
    if rank <= 0:
        try:
            final_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
            with atomic_output_dir(final_dir) as save_dir:
                weight_path = os.path.join(save_dir, WEIGHT_FILENAME)
                save_file(cpu_state, weight_path)
                config_dict = _config_to_dict(transformer)
                config_path = os.path.join(save_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=4)

                if mean_grad_norm is not None:
                    with open(
                        os.path.join(save_dir, "mean_grad_norm.json"),
                        "w",
                    ) as f:
                        json.dump({"mean_grad_norm": mean_grad_norm}, f, indent=4)

                training_state = {
                    "step": int(step),
                    "epoch": int(epoch),
                    "lr_scheduler": (
                        lr_scheduler.state_dict()
                        if lr_scheduler is not None
                        else None
                    ),
                    "has_optimizer_state": optim_state is not None,
                }
                torch.save(
                    training_state,
                    os.path.join(save_dir, TRAINING_STATE_FILENAME),
                )

                if optim_state is not None:
                    torch.save(
                        optim_state,
                        os.path.join(save_dir, OPTIMIZER_FILENAME),
                    )
        except Exception as exc:
            save_error = f"{type(exc).__name__}: {exc}"
    sync_rank0_save_error(rank, save_error, "Transformer checkpoint save")
    main_print(f"--> checkpoint saved at step {step}")


def find_latest_checkpoint(output_dir):
    """Return the `checkpoint-{step}-{epoch}` dir with the highest step, if any."""
    if not output_dir or not os.path.isdir(output_dir):
        return None
    latest = None
    for name in os.listdir(output_dir):
        match = _CHECKPOINT_DIR_RE.match(name)
        if match is None:
            continue
        candidate = os.path.join(output_dir, name)
        if not os.path.isfile(os.path.join(candidate, WEIGHT_FILENAME)):
            continue
        step = int(match.group(1))
        if latest is None or step > latest[0]:
            latest = (step, candidate)
    return latest[1] if latest is not None else None


def resolve_resume_checkpoint(resume_from_checkpoint, output_dir):
    """Turn a `--resume_from_checkpoint` value into a concrete directory."""
    if not resume_from_checkpoint:
        return None
    if str(resume_from_checkpoint).strip().lower() == "latest":
        latest = find_latest_checkpoint(output_dir)
        if latest is None:
            main_print(
                "--> resume_from_checkpoint='latest' but no checkpoint was found "
                f"under {output_dir}; starting from scratch"
            )
        return latest
    path = os.path.expanduser(str(resume_from_checkpoint))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"resume_from_checkpoint directory not found: {path}"
        )
    if not os.path.isfile(os.path.join(path, WEIGHT_FILENAME)):
        raise FileNotFoundError(
            f"{path} does not contain {WEIGHT_FILENAME}; it is not a FlowBP checkpoint"
        )
    return path


def read_training_state(ckpt_dir):
    """Read the small resume metadata, falling back to the directory name."""
    state_path = os.path.join(ckpt_dir, TRAINING_STATE_FILENAME)
    if os.path.isfile(state_path):
        return torch.load(state_path, map_location="cpu", weights_only=False)

    match = _CHECKPOINT_DIR_RE.match(os.path.basename(os.path.normpath(ckpt_dir)))
    step = int(match.group(1)) if match else 0
    epoch = int(match.group(2)) if match else 0
    main_print(
        f"--> {TRAINING_STATE_FILENAME} missing in {ckpt_dir}; "
        f"inferring step={step} from the directory name and resetting the LR schedule"
    )
    return {
        "step": step,
        "epoch": epoch,
        "lr_scheduler": None,
        "has_optimizer_state": False,
    }


def ema_checkpoint_dir(ckpt_dir):
    """Map `.../checkpoint-{s}-{e}` to its sibling `.../checkpoint-ema-{s}-{e}`."""
    normalized = os.path.normpath(ckpt_dir)
    base = os.path.basename(normalized)
    if not _CHECKPOINT_DIR_RE.match(base):
        return None
    ema_dir = os.path.join(
        os.path.dirname(normalized),
        base.replace("checkpoint-", "checkpoint-ema-", 1),
    )
    return ema_dir if os.path.isdir(ema_dir) else None


def load_transformer_weights(model, ckpt_dir, label="transformer"):
    """Load FULL_STATE_DICT weights into a plain module.

    Must be called before FSDP wrapping: the saved keys have no FSDP prefix and
    every rank already holds the unsharded module at that point, so this adds no
    peak memory over the initial `from_pretrained`.
    """
    weight_path = os.path.join(ckpt_dir, WEIGHT_FILENAME)
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"{weight_path} not found")
    state_dict = load_file(weight_path)
    model.load_state_dict(state_dict)
    main_print(f"--> restored {label} weights from {weight_path}")
    return model


def load_optimizer_state(transformer, optimizer, ckpt_dir, rank):
    """Reshard a saved full optimizer state back onto this FSDP process group.

    Returns True when state was restored. Only rank 0 reads the file; FSDP
    scatters the shards to the other ranks.
    """
    optim_path = os.path.join(ckpt_dir, OPTIMIZER_FILENAME)
    if not os.path.isfile(optim_path):
        return False
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state = (
            torch.load(optim_path, map_location="cpu", weights_only=False)
            if rank <= 0
            else {}
        )
        sharded_state = FSDP.optim_state_dict_to_load(
            model=transformer,
            optim=optimizer,
            optim_state_dict=full_state,
        )
    optimizer.load_state_dict(sharded_state)
    main_print(f"--> restored optimizer state from {optim_path}")
    return True


def resume_training_state(transformer, optimizer, lr_scheduler, ckpt_dir, rank):
    """Restore optimizer/LR state after FSDP wrapping and return the resume step.

    Transformer and EMA weights are restored separately, before FSDP wrapping,
    via `load_transformer_weights`.
    """
    state = read_training_state(ckpt_dir)
    init_steps = int(state.get("step", 0))

    if not load_optimizer_state(transformer, optimizer, ckpt_dir, rank):
        main_print(
            f"--> no {OPTIMIZER_FILENAME} in {ckpt_dir}; AdamW moments restart "
            "from zero (use --save_optimizer_state to restore optimizer moments)"
        )

    lr_state = state.get("lr_scheduler")
    if lr_state is not None and lr_scheduler is not None:
        lr_scheduler.load_state_dict(lr_state)
        main_print("--> restored LR scheduler state")
    elif lr_scheduler is not None and init_steps > 0:
        # Replay the schedule so the LR matches the step we are resuming at.
        for _ in range(init_steps):
            lr_scheduler.step()
        main_print(f"--> fast-forwarded LR scheduler by {init_steps} steps")

    main_print(f"--> resuming from {ckpt_dir} at step {init_steps}")
    return init_steps


def save_peft_checkpoint(
    transformer, rank, output_dir, step, epoch, adapter_type="peft"
):
    """Save a standard PEFT adapter from an FSDP-wrapped diffusion model."""
    adapter_type = str(adapter_type).lower()
    main_print(f"--> saving {adapter_type.upper()} checkpoint at step {step}")
    peft_model = getattr(transformer, "module", None)
    if peft_model is None or not hasattr(peft_model, "peft_config"):
        raise TypeError("Expected an FSDP-wrapped PEFT model for checkpointing.")

    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state_dict = transformer.state_dict()

    save_error = None
    if rank <= 0:
        try:
            final_dir = os.path.join(
                output_dir, f"{adapter_type}-checkpoint-{step}-{epoch}"
            )
            with atomic_output_dir(final_dir) as save_dir:
                if adapter_type == "loraga":
                    _save_portable_loraga_checkpoint(
                        peft_model,
                        full_state_dict,
                        save_dir,
                        step,
                        epoch,
                    )
                else:
                    peft_model.save_pretrained(
                        save_dir,
                        state_dict=full_state_dict,
                        safe_serialization=True,
                        is_main_process=True,
                    )
                    with open(
                        os.path.join(save_dir, "training_state.json"),
                        "w",
                    ) as f:
                        json.dump(
                            {
                                "step": step,
                                "epoch": epoch,
                                "adapter_type": adapter_type,
                            },
                            f,
                            indent=4,
                        )
        except Exception as exc:
            save_error = f"{type(exc).__name__}: {exc}"
    sync_rank0_save_error(
        rank,
        save_error,
        f"{adapter_type.upper()} checkpoint save",
    )
    main_print(f"--> {adapter_type.upper()} checkpoint saved at step {step}")


def save_checkpoint_generator_discriminator(
    model,
    optimizer,
    discriminator,
    discriminator_optimizer,
    rank,
    output_dir,
    step,
):
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = model.state_dict()

    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    hf_weight_dir = os.path.join(save_dir, "hf_weights")
    os.makedirs(hf_weight_dir, exist_ok=True)
    if rank <= 0:
        config_dict = _config_to_dict(model)
        config_path = os.path.join(hf_weight_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        weight_path = os.path.join(hf_weight_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)

    main_print(f"--> saved HF weight checkpoint at path {hf_weight_dir}")
    model_weight_dir = os.path.join(save_dir, "model_weights_state")
    os.makedirs(model_weight_dir, exist_ok=True)
    model_optimizer_dir = os.path.join(save_dir, "model_optimizer_state")
    os.makedirs(model_optimizer_dir, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        optim_state = FSDP.optim_state_dict(model, optimizer)
        model_state = model.state_dict()
        weight_state_dict = {"model": model_state}
        dist_cp.save_state_dict(
            state_dict=weight_state_dict,
            storage_writer=dist_cp.FileSystemWriter(model_weight_dir),
            planner=DefaultSavePlanner(),
        )
        optimizer_state_dict = {"optimizer": optim_state}
        dist_cp.save_state_dict(
            state_dict=optimizer_state_dict,
            storage_writer=dist_cp.FileSystemWriter(model_optimizer_dir),
            planner=DefaultSavePlanner(),
        )

    discriminator_fsdp_state_dir = os.path.join(save_dir,
                                                "discriminator_fsdp_state")
    os.makedirs(discriminator_fsdp_state_dir, exist_ok=True)
    with FSDP.state_dict_type(
            discriminator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        optim_state = FSDP.optim_state_dict(discriminator,
                                            discriminator_optimizer)
        model_state = discriminator.state_dict()
        state_dict = {"optimizer": optim_state, "model": model_state}
        if rank <= 0:
            discriminator_fsdp_state_fil = os.path.join(
                discriminator_fsdp_state_dir, "discriminator_state.pt")
            torch.save(state_dict, discriminator_fsdp_state_fil)

    main_print("--> saved FSDP state checkpoint")

def save_lora_checkpoint(transformer, optimizer, rank, output_dir, step,
                         pipeline, epoch):
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state_dict = transformer.state_dict()
        lora_optim_state = FSDP.optim_state_dict(
            transformer,
            optimizer,
        )

    if rank <= 0:
        save_dir = os.path.join(output_dir, f"lora-checkpoint-{step}-{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        # save optimizer
        optim_path = os.path.join(save_dir, "lora_optimizer.pt")
        torch.save(lora_optim_state, optim_path)
        # save lora weight
        main_print(f"--> saving LoRA checkpoint at step {step}")
        transformer_lora_layers = get_peft_model_state_dict(
            model=transformer, state_dict=full_state_dict)
        pipeline.save_lora_weights(
            save_directory=save_dir,
            transformer_lora_layers=transformer_lora_layers,
            is_main_process=True,
        )
        # save config
        lora_config = {
            "step": step,
            "lora_params": {
                "lora_rank": transformer.config.lora_rank,
                "lora_alpha": transformer.config.lora_alpha,
                "target_modules": transformer.config.lora_target_modules,
            },
        }
        config_path = os.path.join(save_dir, "lora_config.json")
        with open(config_path, "w") as f:
            json.dump(lora_config, f, indent=4)
    main_print(f"--> LoRA checkpoint saved at step {step}")
