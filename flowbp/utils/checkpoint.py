import json
import os

import torch
import torch.distributed.checkpoint as dist_cp
from peft import get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.fsdp import (FullOptimStateDictConfig,
                                    FullStateDictConfig)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from flowbp.utils.logging_ import main_print


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


def save_checkpoint(transformer, rank, output_dir, step, epoch, mean_grad_norm=None):
    main_print(f"--> saving checkpoint at step {step}")
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = transformer.state_dict()
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        weight_path = os.path.join(save_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = _config_to_dict(transformer)
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        
        if mean_grad_norm is not None:
            with open(os.path.join(save_dir, 'mean_grad_norm.json'), 'w') as f:
                json.dump({'mean_grad_norm': mean_grad_norm}, f, indent=4)
    main_print(f"--> checkpoint saved at step {step}")


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

    if rank <= 0:
        save_dir = os.path.join(
            output_dir, f"{adapter_type}-checkpoint-{step}-{epoch}"
        )
        os.makedirs(save_dir, exist_ok=True)
        peft_model.save_pretrained(
            save_dir,
            state_dict=full_state_dict,
            safe_serialization=True,
            is_main_process=True,
        )
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            json.dump(
                {"step": step, "epoch": epoch, "adapter_type": adapter_type},
                f,
                indent=4,
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
