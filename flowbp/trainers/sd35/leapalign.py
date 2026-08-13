from __future__ import annotations

import os
import time
from collections import deque
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils import check_min_version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from flowbp.dataset.latent_sd35_rl_datasets import (
    LatentSD35Dataset,
    latent_sd35_collate_function,
)
from flowbp.eval.runners.online import run_online_sd35_eval
from flowbp.rewards.differentiable import (
    compute_differentiable_reward,
    init_differentiable_reward,
    reward_loss_values,
    reward_model_type,
)
from flowbp.schedulers.fm_euler_discrete_with_pred_x0 import (
    FlowMatchEulerDiscreteWithPredX0Scheduler,
)
from flowbp.trainers.common.connector_logging import (
    CONNECTOR_IMAGE_KEYS,
    ENDPOINT_IMAGE_KEYS,
    images_to_wandb_arrays,
    maybe_log_connector_wandb,
    maybe_save_connector_dump,
)
from flowbp.trainers.common.flowbp_lagrange_utils import (
    _jk_histogram_buffer,
    save_jk_sampling_csv,
)
from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.utils.checkpoint import (
    ema_checkpoint_dir,
    load_transformer_weights,
    resolve_resume_checkpoint,
    resume_training_state,
    save_checkpoint,
    save_peft_checkpoint,
)
from flowbp.utils.debug_utils import setup_debug
from flowbp.utils.ema_utils import save_sharded_ema_checkpoint
from flowbp.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from flowbp.utils.gradlora_adapter import (
    GradLoRACalibrator,
    apply_gradlora_if_enabled,
)
from flowbp.utils.logging_ import main_print
from flowbp.utils.loraga_adapter import (
    LoRAGACalibrator,
    apply_loraga_if_enabled,
)
from flowbp.utils.lorapro_optimizer import (
    LoRAProOptimizer,
    discover_peft_lora_pairs,
)
from flowbp.utils.peft_adapter import (
    apply_lora_if_enabled,
    apply_oft_if_enabled,
)
from flowbp.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
)

check_min_version("0.32.0")


def calculate_shift(
    image_seq_len,
    base_seq_len=256,
    max_seq_len=4096,
    base_shift=0.5,
    max_shift=1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _iter_sd35_loader(dataloader, device):
    while True:
        for data_item in dataloader:
            (
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                caption,
            ) = data_item
            yield (
                prompt_embeds.to(device),
                pooled_prompt_embeds.to(device),
                negative_prompt_embeds.to(device),
                negative_pooled_prompt_embeds.to(device),
                caption,
            )
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(dataloader.sampler.epoch + 1)


@torch.no_grad()
def _prepare_lorapro_ignored_parameters(model, device):
    """Keep complete LoRA-Pro factors outside FSDP and synchronize rank 0."""
    target_device = torch.device("cuda", int(device))
    parameters = []
    seen = set()
    for pair in discover_peft_lora_pairs(model):
        for parameter in (pair.a, pair.b):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameter.data = parameter.data.to(target_device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(parameter.data, src=0)
            parameters.append(parameter)
    return parameters


def _save_lorapro_optimizer_shard(optimizer, output_dir, step, epoch, rank):
    save_dir = os.path.join(
        output_dir,
        f"lorapro-optimizer-checkpoint-{step}-{epoch}",
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        optimizer.state_dict(),
        os.path.join(save_dir, f"rank-{rank}.pt"),
    )


def _sample_initial_latents(args, transformer, device, batch_size):
    spatial_downsample = 8
    in_channels = int(getattr(transformer.module if isinstance(transformer, FSDP) else transformer, "config").in_channels)
    latent_h = args.h // spatial_downsample
    latent_w = args.w // spatial_downsample
    latents = torch.randn(
        (batch_size, in_channels, latent_h, latent_w),
        device=device,
        dtype=torch.bfloat16,
    )
    return latents, latent_h, latent_w


def _prepare_timesteps(args, fm_scheduler, transformer, latents, device):
    sample_steps = int(getattr(args, "rollout_steps", None) or args.sampling_steps or 25)
    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)

    # SD3.5 official schedulers ship with use_dynamic_shifting=False (static shift=3.0).
    # Only pass `mu` when the scheduler actually uses dynamic shifting, mirroring the
    # diffusers SD3 pipeline so we do not silently feed an ignored kwarg.
    scheduler_kwargs = {}
    if fm_scheduler.config.get("use_dynamic_shifting", False):
        model = transformer.module if isinstance(transformer, FSDP) else transformer
        patch_size = int(getattr(model.config, "patch_size", 2))
        image_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
        scheduler_kwargs["mu"] = calculate_shift(
            image_seq_len,
            fm_scheduler.config.get("base_image_seq_len", 256),
            fm_scheduler.config.get("max_image_seq_len", 4096),
            fm_scheduler.config.get("base_shift", 0.5),
            fm_scheduler.config.get("max_shift", 1.15),
        )

    timesteps, num_inference_steps = retrieve_timesteps(
        fm_scheduler,
        sample_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    assert num_inference_steps == sample_steps
    return timesteps


def _should_apply_cfg(args, negative_prompt_embeds, negative_pooled_prompt_embeds) -> bool:
    """Single source of truth for whether to enable CFG this step.

    Used by both ``_model_forward_sd35`` (forward graph shape) and
    ``train_sd35_one_step`` (backward loss compensation) so the two stay in
    perfect sync.
    """
    return (
        negative_prompt_embeds is not None
        and negative_pooled_prompt_embeds is not None
        and float(getattr(args, "cfg_guidance", 1.0)) > 1.0
    )


def _model_forward_sd35(
    args,
    transformer,
    latents,
    step_t,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    detach_negative_branch_override=None,
):
    """SD3.5 transformer forward with classifier-free guidance.

    Unlike Flux (guidance distillation, scalar ``guidance`` embedding) SD3.5
    uses real CFG. The default training path runs two separate forwards:
    conditional with autograd, negative under ``torch.no_grad()`` with detached
    latents. Configs can disable that detach/no_grad path via
    ``cfg_detach_neg: false``.

    The combined output ``f = pred_neg + g * (pred_cond - pred_neg)`` is
    numerically identical to the diffusers SD3.5 sampler. The default backward
    path divides the loss by ``g (= cfg_guidance)`` to keep optimizer step size
    roughly invariant to guidance scale; this can be disabled with
    ``cfg_grad_norm_compensate: false``.
    """
    batch_size = latents.shape[0]
    timestep = step_t.expand(batch_size).to(latents.dtype)

    with torch.autocast("cuda", torch.bfloat16):
        pred_cond = transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    if not _should_apply_cfg(args, negative_prompt_embeds, negative_pooled_prompt_embeds):
        return pred_cond

    detach_negative_branch = (
        bool(detach_negative_branch_override)
        if detach_negative_branch_override is not None
        else bool(getattr(args, "cfg_detach_neg", True))
    )
    neg_latents = latents.detach() if detach_negative_branch else latents

    if detach_negative_branch:
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                pred_neg = transformer(
                    hidden_states=neg_latents,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    pooled_projections=negative_pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
    else:
        with torch.autocast("cuda", torch.bfloat16):
            pred_neg = transformer(
                hidden_states=neg_latents,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                pooled_projections=negative_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
    return pred_neg + float(args.cfg_guidance) * (pred_cond - pred_neg)


def _decode_sd35_latents(args, vae, latents):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(vae.config, "shift_factor", 0.0)
        latents = (latents / scaling_factor) + shift_factor
        image_pred_x0 = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        image_pred_x0 = (image_pred_x0 * 0.5 + 0.5).clamp(0, 1)
    return image_pred_x0


def _maybe_decode_sd35_for_wandb(args, vae, latents, max_samples):
    if latents is None:
        return []
    n = min(int(max_samples), latents.shape[0])
    if n <= 0:
        return []
    images = _decode_sd35_latents(args, vae, latents[:n].to(vae.device))
    return images_to_wandb_arrays(images)


def _maybe_log_connector_wandb(
    args,
    vae,
    prefix,
    payload,
    keys=CONNECTOR_IMAGE_KEYS,
):
    maybe_log_connector_wandb(
        args,
        prefix,
        payload,
        lambda latents, max_samples: _maybe_decode_sd35_for_wandb(
            args, vae, latents, max_samples
        ),
        keys=keys,
    )


def _maybe_save_connector_dump(args, prefix, payload):
    maybe_save_connector_dump(args, prefix, payload)


def _optional_positive_float(value):
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _build_traj_filter_mask(args, d_j: torch.Tensor, d_0: torch.Tensor):
    """Return keep mask for per-sample trajectory filtering, or None if disabled."""
    dj_max = _optional_positive_float(getattr(args, "traj_filter_dj_max", None))
    d0_max = _optional_positive_float(getattr(args, "traj_filter_d0_max", None))
    if dj_max is None and d0_max is None:
        args._last_sample_keep_mask = None
        return None, {}

    keep_mask = torch.ones_like(d_j, dtype=torch.bool)
    if dj_max is not None:
        keep_mask = keep_mask & (d_j <= dj_max)
    if d0_max is not None:
        keep_mask = keep_mask & (d_0 <= d0_max)

    args._last_sample_keep_mask = keep_mask.detach()
    keep_rate = keep_mask.float().mean()
    return keep_mask, {
        "traj_filter_enabled": torch.tensor(1.0, device=d_j.device, dtype=torch.float32),
        "traj_filter_keep_rate": keep_rate.detach(),
        "traj_filter_drop_rate": (1.0 - keep_rate).detach(),
        "traj_filter_kept": keep_mask.float().sum().detach(),
    }


def sample_trajectory_sd35(
    args,
    device,
    transformer,
    fm_scheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    generator,
):
    args._last_sample_keep_mask = None
    latents_xt, _, _ = _sample_initial_latents(args, transformer, device, prompt_embeds.shape[0])
    timesteps = _prepare_timesteps(args, fm_scheduler, transformer, latents_xt, device)
    select_indices, k_idx, j_idx = sample_jk_indices(
        args, timesteps.size(0), generator
    )
    jk_truncated = bool(getattr(args, "_last_jk_truncated", False))
    grad_on_ind_set = set(
        timesteps.size(0) - select_index.item()
        for select_index in select_indices
    )

    x_j_reference = None
    pred_xt2 = None
    latents_pred_x0 = None
    for idx, step_t in enumerate(timesteps):
        with nullcontext() if (idx in grad_on_ind_set) else torch.no_grad():
            if idx == timesteps.size(0) - select_indices[1].item():
                with torch.no_grad():
                    pred_x2_raw = torch.abs(
                        pred_xt2.double() - latents_xt.double()
                    ).mean(
                        dim=tuple(range(1, pred_xt2.ndim)),
                        keepdim=False,
                    )
                    pred_x2_weight_factor = pred_x2_raw.clip(min=args.tau)
                # Match Flux LeapAlign: stop grads through the Euler branch while keeping the jump branch for theta.
                latents_xt = pred_xt2 + (latents_xt - pred_xt2).detach()
                if args.alpha <= 0.0:
                    dit_latents_xt = latents_xt.detach()
                else:
                    dit_latents_xt = (
                        args.alpha * latents_xt
                        + (1 - args.alpha) * latents_xt.detach()
                    )
            else:
                dit_latents_xt = latents_xt

            pred = _model_forward_sd35(
                args,
                transformer,
                dit_latents_xt,
                step_t,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            )

            if idx == timesteps.size(0) - select_indices[0].item():
                t2_idx = timesteps.size(0) - select_indices[1].item()
                pred_xt2 = fm_scheduler.jump_to_step(pred, step_t, latents_xt, t2_idx)
                pred = pred.detach()
            if idx == timesteps.size(0) - select_indices[1].item():
                x_j_reference = latents_xt.detach()
                latents_xt, latents_pred_x0 = fm_scheduler.step(
                    pred,
                    step_t,
                    latents_xt,
                    with_pred_x0=True,
                )
                latents_xt = latents_xt.detach()
            else:
                latents_xt = fm_scheduler.step(
                    pred,
                    step_t,
                    latents_xt,
                    with_pred_x0=False,
                )[0]
            if latents_xt.dtype != torch.bfloat16:
                latents_xt = latents_xt.to(torch.bfloat16)

    with torch.no_grad():
        pred_x0_raw = torch.abs(latents_pred_x0.double() - latents_xt.double()).mean(
            dim=tuple(range(1, latents_pred_x0.ndim)), keepdim=False
        )
        pred_x0_weight_factor = pred_x0_raw.clip(min=args.tau)
        traj_dist = pred_x2_weight_factor + pred_x0_weight_factor
        traj_sim_weight_factor = 1.0 / traj_dist
        traj_metrics = {
            "d0": pred_x0_raw.detach().mean(),
            "dj": pred_x2_raw.detach().mean(),
            "d0_clipped": pred_x0_weight_factor.detach().mean(),
            "dj_clipped": pred_x2_weight_factor.detach().mean(),
            "d_sum": traj_dist.detach().mean(),
            "w_sim": traj_sim_weight_factor.detach().mean(),
        }
    traj_metrics.update({
        "j_idx": torch.tensor(float(j_idx), device=latents_xt.device, dtype=torch.float32),
        "k_idx": torch.tensor(float(k_idx), device=latents_xt.device, dtype=torch.float32),
        "jk_gap": torch.tensor(float(j_idx - k_idx), device=latents_xt.device, dtype=torch.float32),
        "j_rev": torch.tensor(float(timesteps.size(0) - j_idx), device=latents_xt.device, dtype=torch.float32),
        "k_rev": torch.tensor(float(timesteps.size(0) - k_idx), device=latents_xt.device, dtype=torch.float32),
        "jk_truncated": torch.tensor(1.0 if jk_truncated else 0.0, device=latents_xt.device, dtype=torch.float32),
    })

    latents_pred_x0 = latents_pred_x0 + (latents_xt - latents_pred_x0).detach()
    if x_j_reference is not None:
        d_j_dump = torch.abs(pred_xt2.double() - x_j_reference.double()).mean(
            dim=tuple(range(1, pred_xt2.ndim)),
            keepdim=False,
        )
        d_0_dump = torch.abs(latents_pred_x0.double() - latents_xt.double()).mean(
            dim=tuple(range(1, latents_pred_x0.ndim)),
            keepdim=False,
        )
        _, filter_metrics = _build_traj_filter_mask(args, d_j_dump, d_0_dump)
        traj_metrics.update(filter_metrics)
        connector_payload = {
            "step": getattr(args, "current_train_step", None),
            "k_idx": timesteps.size(0) - select_indices[0].item(),
            "j_idx": timesteps.size(0) - select_indices[1].item(),
            "select_indices": select_indices.detach(),
            "xj_pred": pred_xt2.detach(),
            "xj": x_j_reference.detach(),
            "x0_pred": latents_pred_x0.detach(),
            "x0": latents_xt.detach(),
            "dj_per_sample": d_j_dump.detach(),
            "d0_per_sample": d_0_dump.detach(),
            "weight_scheme": "euler_jump",
        }
        _maybe_save_connector_dump(args, "sd35_leapalign", connector_payload)
        args._last_connector_payload = connector_payload
    return latents_xt, latents_pred_x0, traj_sim_weight_factor, traj_metrics


def _repeat_captions(caption, repeats: int):
    if repeats <= 1:
        return caption
    if isinstance(caption, str):
        return [caption] * repeats
    return list(caption) * repeats


def _slice_captions(caption, batch_size: int):
    if isinstance(caption, str):
        return caption
    return list(caption)[:batch_size]


def get_reward_loss_sd35(
    args,
    latents_pred_x0,
    sample_weights,
    sample_keep_mask,
    vae,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
):
    image_pred_x0 = _decode_sd35_latents(args, vae, latents_pred_x0)
    reward_pred_x0 = compute_differentiable_reward(
        args,
        image_pred_x0,
        caption,
        reward_model,
        tokenizer,
        preprocess_val,
    )
    raw_loss = reward_loss_values(args, reward_pred_x0)
    if sample_weights is not None:
        sample_weights = sample_weights.to(raw_loss.device, raw_loss.dtype)
        raw_loss = sample_weights * raw_loss
    if sample_keep_mask is not None:
        sample_keep_mask = sample_keep_mask.to(device=raw_loss.device, dtype=torch.bool)
        if sample_keep_mask.any():
            raw_loss = raw_loss[sample_keep_mask]
        else:
            raw_loss = raw_loss.sum().view(()) * 0.0
    return torch.mean(raw_loss), reward_pred_x0.detach().mean()


def _align_grad_dtype_to_param_dtype(model):
    """AdamW expects each param's grad to match the param/state dtype.

    FSDP mixed precision can leave a few gradients in bf16 while the optimizer
    state for the corresponding flat parameter is fp32 (or vice versa). The
    single-tensor AdamW path then fails at exp_avg.lerp_(grad, ...). Casting
    after clipping keeps optimizer math well-typed without changing the forward
    graph.
    """
    for param in model.parameters():
        if param.grad is not None and param.grad.dtype != param.dtype:
            param.grad.data = param.grad.data.to(param.dtype)


def train_sd35_one_step(
    args,
    inner_step,
    device,
    transformer,
    vae,
    fm_scheduler,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    preprocess_val,
    select_idx_generator,
    perform_optimizer_step=True,
    trainer_override=None,
    batch_size_override=None,
):
    (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        caption,
    ) = next(loader)
    if batch_size_override is not None:
        effective_batch_size = int(batch_size_override)
        if effective_batch_size < 1:
            raise ValueError(
                f"batch_size_override must be at least 1, got {effective_batch_size}"
            )
        available_batch_size = int(prompt_embeds.shape[0])
        effective_batch_size = min(effective_batch_size, available_batch_size)
        prompt_embeds = prompt_embeds[:effective_batch_size]
        pooled_prompt_embeds = pooled_prompt_embeds[:effective_batch_size]
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds[:effective_batch_size]
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds[
                :effective_batch_size
            ]
        caption = _slice_captions(caption, effective_batch_size)
    caption_for_reward = caption
    effective_trainer = trainer_override or args.trainer
    # Function-local imports keep the per-algorithm trainer modules decoupled
    # from this module at import time (they import shared backend helpers from
    # here), avoiding circular imports while still dispatching by trainer name.
    from flowbp.trainers.sd35.flowbp_bridge import sample_flowbp_bridge_trajectory_sd35
    from flowbp.trainers.sd35.flowbp_sparse import sample_flowbp_sparse_trajectory_sd35
    from flowbp.trainers.sd35.draft_lv import sample_draft_lv_trajectory_sd35
    from flowbp.trainers.sd35.drtune import sample_drtune_trajectory_sd35
    from flowbp.trainers.sd35.fullbp import sample_fullbp_trajectory_sd35
    from flowbp.trainers.sd35.flowbp_lagrange import sample_flowbp_lagrange_trajectory_sd35
    from flowbp.trainers.sd35.refl import sample_refl_trajectory_sd35

    if effective_trainer == "fullbp":
        latents_pred_x0, sample_weights, traj_metrics = (
            sample_fullbp_trajectory_sd35(
                args,
                device,
                transformer,
                fm_scheduler,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                select_idx_generator,
            )
        )
    elif effective_trainer == "flowbp_lagrange":
        latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_lagrange_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        _maybe_log_connector_wandb(args, vae, "sd35_flowbp_lagrange", getattr(args, "_last_connector_payload", None))
    elif effective_trainer == "draft_lv":
        _, latents_pred_x0, sample_weights, traj_metrics = sample_draft_lv_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        repeat_factor = latents_pred_x0.shape[0] // prompt_embeds.shape[0]
        caption_for_reward = _repeat_captions(caption, repeat_factor)
    elif effective_trainer == "drtune":
        _, latents_pred_x0, sample_weights, traj_metrics = sample_drtune_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
    elif effective_trainer == "refl":
        _, latents_pred_x0, sample_weights, traj_metrics = sample_refl_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
    elif effective_trainer == "flowbp_sparse":
        latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_sparse_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        _maybe_log_connector_wandb(
            args,
            vae,
            "sd35_flowbp_sparse",
            getattr(args, "_last_connector_payload", None),
            keys=ENDPOINT_IMAGE_KEYS,
        )
    elif effective_trainer == "flowbp_bridge":
        latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_bridge_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        _maybe_log_connector_wandb(args, vae, "sd35_flowbp_bridge", getattr(args, "_last_connector_payload", None))
    else:
        _, latents_pred_x0, sample_weights, traj_metrics = sample_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        _maybe_log_connector_wandb(args, vae, "sd35_leapalign", getattr(args, "_last_connector_payload", None))

    if effective_trainer == "flowbp_lagrange":
        payload = getattr(args, "_last_connector_payload", None)
        current_step = getattr(args, "current_train_step", None)
        if payload is not None and current_step is not None and dist.get_rank() == 0:
            _jk_histogram_buffer["k_idx"].append(float(payload["k_idx"]))
            _jk_histogram_buffer["j_idx"].append(float(payload["j_idx"]))
            _jk_histogram_buffer["jk_gap"].append(float(payload["j_idx"] - payload["k_idx"]))
            total = float(getattr(args, "sampling_steps", 25) or 25)
            _jk_histogram_buffer.setdefault("seg_a", []).append(total - float(payload["j_idx"]))
            _jk_histogram_buffer.setdefault("seg_b", []).append(float(payload["j_idx"] - payload["k_idx"]))
            _jk_histogram_buffer.setdefault("seg_c", []).append(float(payload["k_idx"]))

    loss, avg_reward_pred_x0 = get_reward_loss_sd35(
        args,
        latents_pred_x0,
        sample_weights,
        getattr(args, "_last_sample_keep_mask", None),
        vae,
        reward_model,
        tokenizer,
        caption_for_reward,
        preprocess_val,
    )
    # Capture the unscaled loss before applying CFG gradient compensation, so
    # the metric remains comparable across guidance scales.
    avg_loss = loss.detach()

    # CFG gradient compensation. By default this cancels the linear gradient
    # amplification from CFG, so guidance changes mostly affect the forward
    # rollout rather than the optimizer step size.
    cfg_compensation = float(getattr(args, "cfg_guidance", 1.0))
    use_cfg_compensation = bool(getattr(args, "cfg_grad_norm_compensate", True))
    backward_divisor = float(args.gradient_accumulation_steps) * (
        cfg_compensation
        if (
            use_cfg_compensation
            and _should_apply_cfg(args, negative_prompt_embeds, negative_pooled_prompt_embeds)
        )
        else 1.0
    )
    (loss / backward_divisor).backward()
    dist.all_reduce(avg_loss.div_(args.gradient_accumulation_steps), op=dist.ReduceOp.AVG)
    dist.all_reduce(avg_reward_pred_x0.div_(args.gradient_accumulation_steps), op=dist.ReduceOp.AVG)
    reduced_traj_metrics = {}
    for metric_name, metric_value in traj_metrics.items():
        if metric_value is None:
            reduced_traj_metrics[metric_name] = None
            continue
        metric_tensor = metric_value.detach().float().to(device)
        dist.all_reduce(metric_tensor.div_(args.gradient_accumulation_steps), op=dist.ReduceOp.AVG)
        reduced_traj_metrics[metric_name] = metric_tensor.item()

    if bool(getattr(args, "gradlora_freeze_a", False)):
        # A is intentionally kept requires_grad=True under FSDP, but it must
        # neither accumulate across steps nor contribute to global norm
        # clipping. B gradients are unaffected by clearing A after backward.
        for name, parameter in transformer.named_parameters():
            if ".lora_A." in name and parameter.grad is not None:
                # Zero the view in place instead of assigning None: with
                # use_orig_params=True this also clears A's segment in the
                # underlying FSDP flat gradient used by clip_grad_norm_.
                parameter.grad.zero_()

    if (
        perform_optimizer_step
        and (inner_step + 1) % args.gradient_accumulation_steps == 0
    ):
        if bool(getattr(args, "use_lora_pro", False)):
            optimizer.step()
            diagnostics = optimizer.get_step_diagnostics()
            grad_norm = float(diagnostics["raw_grad_norm"])
            reduced_traj_metrics.update(
                {
                    f"lorapro_{name}": value
                    for name, value in diagnostics.items()
                    if isinstance(value, (int, float))
                }
            )
        else:
            grad_norm = transformer.clip_grad_norm_(args.max_grad_norm)
            _align_grad_dtype_to_param_dtype(transformer)
            optimizer.step()
            grad_norm = grad_norm.item()
        lr_scheduler.step()
        optimizer.zero_grad()
    else:
        grad_norm = None
    return avg_loss.item(), avg_reward_pred_x0.item(), grad_norm, reduced_traj_metrics


def normalize_sd35_trainer_name(name: str | None) -> str:
    raw_name = str(name or "leapalign").lower().replace("-", "_")
    aliases = {
        "flowbp_sparse": "flowbp_sparse",
        "flowbp_bridge": "flowbp_bridge",
        "flowbp_lagrange": "flowbp_lagrange",
    }
    return aliases.get(raw_name, raw_name)

def run_training_sd35(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    trainer_name = normalize_sd35_trainer_name(getattr(args, "trainer", "leapalign"))
    args.trainer = trainer_name
    if trainer_name not in {
        "fullbp",
        "leapalign",
        "refl",
        "flowbp_lagrange",
        "flowbp_sparse",
        "flowbp_bridge",
        "draft_lv",
        "drtune",
    }:
        raise ValueError(
            "SD3.5 trainer supports FlowBP method names (flowbp_sparse, "
            "flowbp_bridge, flowbp_lagrange) and baselines "
            "(fullbp, leapalign, refl, draft_lv, drtune)."
        )
    if int(getattr(args, "sp_size", 1)) != 1:
        raise NotImplementedError("SD3.5 trainer currently supports sp_size=1 only.")
    if args.h is None or args.w is None or args.h % 16 != 0 or args.w % 16 != 0:
        raise ValueError("SD3.5 training expects h/w to be set and divisible by 16.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.debug:
        setup_debug()
    main_print("------------------args------------------")
    for arg_k, arg_v in vars(args).items():
        main_print(f"{arg_k}: {arg_v}")
    main_print("------------------args------------------")
    if float(getattr(args, "cfg", 0.0)) > 0.0 and float(getattr(args, "cfg_guidance", 1.0)) > 1.0:
        main_print(
            "[warning] SD3.5 RL training usually wants data.cfg=0.0. "
            "With cfg>0, dropped samples become unconditional under CFG."
        )
    if args.seed is not None:
        set_seed(args.seed + rank)
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    requested_adapters = any(
        bool(getattr(args, name, False))
        for name in ("use_lora", "use_gradlora", "use_loraga", "use_oft")
    )
    if getattr(args, "resume_from_checkpoint", None) and requested_adapters:
        raise NotImplementedError(
            "SD3.5 adapter checkpoints cannot currently be resumed. "
            "Refusing to silently restart adapter training."
        )

    selected_reward = reward_model_type(args)
    reward_model, processor, preprocess_val = init_differentiable_reward(
        args, device
    )
    main_print(f"--> differentiable reward: {selected_reward}")

    resume_ckpt_dir = resolve_resume_checkpoint(
        getattr(args, "resume_from_checkpoint", None), args.output_dir
    )

    main_print(f"--> loading SD3 model from {args.pretrained_model_name_or_path}")
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    fm_scheduler = FlowMatchEulerDiscreteWithPredX0Scheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    if resume_ckpt_dir is not None:
        load_transformer_weights(transformer, resume_ckpt_dir)

    enabled_adapters = [
        name
        for name, enabled in (
            ("lora", args.use_lora),
            ("gradlora", args.use_gradlora),
            ("loraga", args.use_loraga),
            ("oft", args.use_oft),
        )
        if enabled
    ]
    if len(enabled_adapters) > 1:
        raise ValueError(
            "LoRA, GradLoRA, LoRA-GA, and OFT are mutually exclusive; "
            "enable only one. "
            f"Enabled: {enabled_adapters}"
        )
    if bool(getattr(args, "use_lora_pro", False)) and not bool(args.use_lora):
        raise ValueError("LoRA-Pro requires lora.enabled=true")
    if bool(getattr(args, "use_lora_pro", False)) and float(args.weight_decay) != 0.0:
        raise ValueError("LoRA-Pro requires optimizer.weight_decay=0")
    transformer, use_lora = apply_lora_if_enabled(transformer, args)
    transformer, use_gradlora = apply_gradlora_if_enabled(transformer, args)
    transformer, use_loraga = apply_loraga_if_enabled(transformer, args)
    transformer, use_oft = apply_oft_if_enabled(transformer, args)
    use_lora_pro = bool(getattr(args, "use_lora_pro", False))
    use_peft = use_lora or use_gradlora or use_loraga or use_oft
    adapter_type = (
        "lorapro"
        if use_lora_pro
        else "lora"
        if use_lora
        else "gradlora"
        if use_gradlora
        else "loraga"
        if use_loraga
        else "oft"
        if use_oft
        else None
    )
    lorapro_ignored_parameters = (
        _prepare_lorapro_ignored_parameters(transformer, device)
        if use_lora_pro
        else None
    )
    ema_model = deepcopy(transformer) if args.use_ema else None
    if ema_model is not None and resume_ckpt_dir is not None:
        ema_dir = ema_checkpoint_dir(resume_ckpt_dir)
        if ema_dir is not None:
            load_transformer_weights(ema_model, ema_dir, label="EMA")
        else:
            main_print(
                "--> no EMA checkpoint alongside the resume directory; "
                "seeding EMA from the restored transformer weights"
            )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        use_peft,
        args.use_cpu_offload,
        args.master_weight_type,
        ignored_states=lorapro_ignored_parameters,
    )
    transformer = FSDP(transformer, **fsdp_kwargs)
    if args.use_ema and ema_model is not None:
        ema_ignored_parameters = (
            _prepare_lorapro_ignored_parameters(ema_model, device)
            if use_lora_pro
            else None
        )
        ema_fsdp_kwargs, _ = get_dit_fsdp_kwargs(
            ema_model,
            args.fsdp_sharding_startegy,
            use_peft,
            args.use_cpu_offload,
            args.master_weight_type,
            ignored_states=ema_ignored_parameters,
        )
        ema_model = FSDP(ema_model, **ema_fsdp_kwargs)
        ema_model.eval()
        ema_model.requires_grad_(False)
    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules, args.selective_checkpointing)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    transformer.train()

    if use_lora_pro:
        optimizer = LoRAProOptimizer(
            transformer,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=float(args.adam_eps),
            weight_decay=0.0,
            damping=float(args.lora_pro_damping),
            rcond=args.lora_pro_rcond,
            max_grad_norm=float(args.max_grad_norm),
            shard_dense_states=bool(args.lora_pro_shard_dense_states),
        )
    else:
        params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=float(args.adam_eps),
            fused=False,
            foreach=False,
        )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=-1,
    )

    init_steps = 0
    if resume_ckpt_dir is not None:
        init_steps = resume_training_state(
            transformer, optimizer, lr_scheduler, resume_ckpt_dir, rank
        )
        if init_steps >= args.max_train_steps:
            main_print(
                f"--> checkpoint step {init_steps} already reaches max_train_steps="
                f"{args.max_train_steps}; nothing to do"
            )

    train_dataset = LatentSD35Dataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_sd35_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    select_idx_generator = (
        torch.Generator(device="cpu").manual_seed(args.select_idx_seed)
        if args.select_idx_seed is not None
        else None
    )
    loader = _iter_sd35_loader(train_dataloader, device)
    if use_loraga:
        calibration_batches = int(args.loraga_calibration_batches)
        if calibration_batches < 1:
            raise ValueError("loraga_calibration_batches must be at least 1")
        calibration_mode = str(args.loraga_calibration_mode).lower()
        if calibration_mode not in {"sparse", "fullbp"}:
            raise ValueError(
                "loraga_calibration_mode must be 'sparse' or 'fullbp'"
            )
        fullbp_batch_size = int(args.loraga_fullbp_batch_size)
        if fullbp_batch_size < 1:
            raise ValueError("loraga_fullbp_batch_size must be at least 1")
        diagnostics_path = args.loraga_diagnostics_path or os.path.join(
            args.output_dir or ".",
            "loraga_calibration.json",
        )
        use_second_moment = bool(
            getattr(args, "loraga_use_second_moment", False)
        )
        second_moment_beta = float(
            getattr(args, "loraga_second_moment_beta", 0.5)
        )
        if use_second_moment and not 0.0 < second_moment_beta <= 1.0:
            raise ValueError(
                "loraga_second_moment_beta must be in (0, 1] when "
                f"loraga_use_second_moment is enabled, got {second_moment_beta}"
            )
        main_print(
            f"***** LoRA-GA calibration: {calibration_batches} "
            f"reward-backprop batches (mode={calibration_mode}, "
            f"gamma={args.loraga_stable_gamma}, "
            f"second_moment={use_second_moment}, "
            f"beta={second_moment_beta if use_second_moment else 0.0}, "
            f"use_rslora={args.loraga_use_rslora}) *****"
        )
        calibrator = LoRAGACalibrator(
            transformer,
            accumulation_device=args.loraga_accumulation_device,
            offload_inputs=calibration_mode == "fullbp",
            use_second_moment=use_second_moment,
            second_moment_beta=second_moment_beta,
            second_moment_rank=int(args.loraga_rank),
            second_moment_oversample=int(args.loraga_svd_oversample),
            moment_compute_device=args.loraga_svd_device,
        )
        optimizer.zero_grad(set_to_none=True)
        args.current_train_step = 0
        nonzero_calibration_batches = 0
        for calibration_step in range(calibration_batches):
            avg_loss, avg_reward, _, _ = train_sd35_one_step(
                args,
                0,
                device,
                transformer,
                vae,
                fm_scheduler,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                loader,
                preprocess_val,
                select_idx_generator,
                perform_optimizer_step=False,
                trainer_override=(
                    "fullbp" if calibration_mode == "fullbp" else None
                ),
                batch_size_override=(
                    fullbp_batch_size if calibration_mode == "fullbp" else None
                ),
            )
            calibrator.finish_batch()
            optimizer.zero_grad(set_to_none=True)
            main_print(
                f"  LoRA-GA calibration batch "
                f"{calibration_step + 1}/{calibration_batches}: "
                f"loss={avg_loss:.4f}, reward={avg_reward:.4f}"
            )
            if abs(float(avg_loss)) >= 1e-12:
                nonzero_calibration_batches += 1
        if nonzero_calibration_batches == 0:
            calibrator.close()
            raise RuntimeError(
                "Every LoRA-GA calibration batch had zero reward loss"
            )
        calibrator.finalize(
            rank=int(args.loraga_rank),
            oversample=int(args.loraga_svd_oversample),
            niter=int(args.loraga_svd_niter),
            svd_device=args.loraga_svd_device,
            diagnostics_path=diagnostics_path,
            metadata={
                "calibration_mode": calibration_mode,
                "calibration_batches": calibration_batches,
                "calibration_batch_size": (
                    fullbp_batch_size
                    if calibration_mode == "fullbp"
                    else int(args.train_batch_size)
                ),
                "fullbp_full_cfg_grad": bool(
                    args.loraga_fullbp_full_cfg_grad
                ),
                "stable_gamma": float(args.loraga_stable_gamma),
                "use_rslora": bool(args.loraga_use_rslora),
                "a_subspace": (
                    "second_moment" if use_second_moment else "mean"
                ),
            },
        )
        optimizer.zero_grad(set_to_none=True)
        if args.use_ema and ema_model is not None:
            with torch.no_grad():
                for target_param, source_param in zip(
                    ema_model.parameters(),
                    transformer.parameters(),
                ):
                    target_param.data.copy_(
                        source_param.data.to(
                            device=target_param.device,
                            dtype=target_param.dtype,
                        )
                    )
        main_print("***** LoRA-GA calibration complete *****")
    if use_gradlora:
        calibration_batches = int(args.gradlora_calibration_batches)
        if calibration_batches < 1:
            raise ValueError(
                "gradlora_calibration_batches must be at least 1, "
                f"got {calibration_batches}"
            )
        calibration_mode = str(
            getattr(args, "gradlora_calibration_mode", "sparse")
        ).lower()
        if calibration_mode not in {"sparse", "fullbp"}:
            raise ValueError(
                "gradlora_calibration_mode must be 'sparse' or 'fullbp', "
                f"got {calibration_mode!r}"
            )
        fullbp_batch_size = int(
            getattr(args, "gradlora_fullbp_batch_size", 1)
        )
        if fullbp_batch_size < 1:
            raise ValueError(
                "gradlora_fullbp_batch_size must be at least 1, "
                f"got {fullbp_batch_size}"
            )
        use_second_moment = bool(
            getattr(args, "gradlora_use_second_moment", False)
        )
        second_moment_beta = float(
            getattr(args, "gradlora_second_moment_beta", 0.5)
        )
        spectral_exponent = float(
            getattr(args, "gradlora_spectral_exponent", 0.0)
        )
        if use_second_moment and not 0.0 < second_moment_beta <= 1.0:
            raise ValueError(
                "gradlora_second_moment_beta must be in (0, 1] when "
                f"second-moment calibration is enabled, got {second_moment_beta}"
            )
        diagnostics_path = args.gradlora_diagnostics_path
        if not diagnostics_path:
            diagnostics_path = os.path.join(
                args.output_dir or ".",
                "gradlora_calibration.json",
            )
        usefulness_batches = int(
            getattr(args, "gradlora_usefulness_batches", 0) or 0
        )
        if usefulness_batches < 0:
            raise ValueError(
                "gradlora_usefulness_batches must be non-negative, "
                f"got {usefulness_batches}"
            )
        usefulness_path = os.path.join(
            args.output_dir or ".",
            "gradlora_usefulness.json",
        )
        init_variant = str(
            getattr(args, "gradlora_init_variant", "selected")
        ).lower()
        if init_variant not in {"selected", "mean", "random"}:
            raise ValueError(
                "gradlora_init_variant must be 'selected', 'mean', or "
                f"'random', got {init_variant!r}"
            )
        a_scale_mode = str(
            getattr(args, "gradlora_a_scale_mode", "legacy")
        ).lower()
        a_scale = float(getattr(args, "gradlora_a_scale", 1.0))
        freeze_a = bool(getattr(args, "gradlora_freeze_a", False))
        main_print(
            f"***** GradLoRA calibration: {calibration_batches} "
            f"reward-backprop batches (mode={calibration_mode}, "
            f"second_moment={use_second_moment}, "
            f"beta={second_moment_beta if use_second_moment else 0.0}, "
            f"spectral_exponent={spectral_exponent}, "
            f"a_scale_mode={a_scale_mode}, a_scale={a_scale}) *****"
        )
        calibrator = GradLoRACalibrator(
            transformer,
            accumulation_device=args.gradlora_accumulation_device,
            offload_inputs=calibration_mode == "fullbp",
            use_second_moment=use_second_moment,
            second_moment_beta=second_moment_beta,
            second_moment_rank=int(args.gradlora_rank),
            second_moment_oversample=int(args.gradlora_svd_oversample),
            moment_compute_device=args.gradlora_svd_device,
            collect_probe_state=usefulness_batches > 0,
            a_scale_mode=a_scale_mode,
            a_scale=a_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        args.current_train_step = 0
        nonzero_calibration_batches = 0
        for calibration_step in range(calibration_batches):
            avg_loss, avg_reward, _, _ = train_sd35_one_step(
                args,
                0,
                device,
                transformer,
                vae,
                fm_scheduler,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                loader,
                preprocess_val,
                select_idx_generator,
                perform_optimizer_step=False,
                trainer_override=(
                    "fullbp" if calibration_mode == "fullbp" else None
                ),
                batch_size_override=(
                    fullbp_batch_size if calibration_mode == "fullbp" else None
                ),
            )
            calibrator.finish_batch()
            optimizer.zero_grad(set_to_none=True)
            main_print(
                f"  GradLoRA calibration batch "
                f"{calibration_step + 1}/{calibration_batches}: "
                f"loss={avg_loss:.4f}, reward={avg_reward:.4f}"
            )
            if abs(float(avg_loss)) < 1e-12:
                main_print(
                    "[warning] GradLoRA calibration loss is ~0 "
                    f"(reward={avg_reward:.4f}). ReLU clip may have zeroed "
                    "all reward gradients; GradLoRA will fall back to random "
                    "A on inactive layers."
                )
            else:
                nonzero_calibration_batches += 1
        if nonzero_calibration_batches == 0:
            calibrator.close()
            raise RuntimeError(
                "Every GradLoRA calibration batch had zero reward loss, so no "
                "reward-gradient direction can be estimated. Adjust "
                "loss_relu_clip or the calibration data before retrying."
            )
        calibrator.finalize(
            rank=int(args.gradlora_rank),
            spectral_exponent=spectral_exponent,
            oversample=int(args.gradlora_svd_oversample),
            niter=int(args.gradlora_svd_niter),
            svd_device=args.gradlora_svd_device,
            init_variant=init_variant,
            diagnostics_path=diagnostics_path,
            metadata={
                "init_variant": init_variant,
                "freeze_a": freeze_a,
                "calibration_mode": calibration_mode,
                "calibration_batches": calibration_batches,
                "use_second_moment": use_second_moment,
                "second_moment_beta": (
                    second_moment_beta if use_second_moment else 0.0
                ),
                "spectral_exponent": spectral_exponent,
                "calibration_batch_size": (
                    fullbp_batch_size
                    if calibration_mode == "fullbp"
                    else int(args.train_batch_size)
                ),
                "fullbp_full_cfg_grad": bool(
                    getattr(args, "gradlora_fullbp_full_cfg_grad", False)
                ),
                "rollout_steps": int(
                    getattr(args, "rollout_steps", None)
                    or args.sampling_steps
                    or 25
                ),
            },
        )
        optimizer.zero_grad(set_to_none=True)
        if usefulness_batches > 0:
            main_print(
                f"***** GradLoRA usefulness probe: {usefulness_batches} "
                "held-out batches *****"
            )
            calibrator.begin_probe()
            for probe_step in range(usefulness_batches):
                avg_loss, avg_reward, _, _ = train_sd35_one_step(
                    args,
                    0,
                    device,
                    transformer,
                    vae,
                    fm_scheduler,
                    reward_model,
                    processor,
                    optimizer,
                    lr_scheduler,
                    loader,
                    preprocess_val,
                    select_idx_generator,
                    perform_optimizer_step=False,
                    trainer_override=(
                        "fullbp" if calibration_mode == "fullbp" else None
                    ),
                    batch_size_override=(
                        fullbp_batch_size
                        if calibration_mode == "fullbp"
                        else None
                    ),
                )
                calibrator.finish_batch()
                optimizer.zero_grad(set_to_none=True)
                main_print(
                    f"  GradLoRA held-out batch "
                    f"{probe_step + 1}/{usefulness_batches}: "
                    f"loss={avg_loss:.4f}, reward={avg_reward:.4f}"
                )
            calibrator.finish_probe(
                diagnostics_path=usefulness_path,
                metadata={
                    "init_variant": init_variant,
                    "calibration_mode": calibration_mode,
                    "calibration_batches": calibration_batches,
                    "use_second_moment": use_second_moment,
                    "second_moment_beta": (
                        second_moment_beta if use_second_moment else 0.0
                    ),
                    "spectral_exponent": spectral_exponent,
                    "rank": int(args.gradlora_rank),
                },
            )
            optimizer.zero_grad(set_to_none=True)
        if freeze_a:
            # FSDP fixes per-flat-parameter requires_grad at wrap time, so drop
            # A from the optimizer instead of flipping requires_grad. Gradients
            # are cleared after backward and never applied to A.
            frozen_parameter_ids: set[int] = set()
            for name, parameter in transformer.named_parameters():
                if parameter.requires_grad and ".lora_A." in name:
                    frozen_parameter_ids.add(id(parameter))
            if not frozen_parameter_ids:
                raise RuntimeError(
                    "Freezing GradLoRA A matched no lora_A parameter names"
                )
            kept = 0
            for group in optimizer.param_groups:
                group["params"] = [
                    parameter
                    for parameter in group["params"]
                    if id(parameter) not in frozen_parameter_ids
                ]
                kept += len(group["params"])
            if kept == 0:
                raise RuntimeError(
                    "Freezing GradLoRA A left no trainable parameter"
                )
            main_print(
                f"--> GradLoRA froze {len(frozen_parameter_ids)} lora_A tensors "
                f"({kept} optimizer tensors remain)"
            )
        if args.use_ema and ema_model is not None:
            with torch.no_grad():
                for target_param, source_param in zip(
                    ema_model.parameters(),
                    transformer.parameters(),
                ):
                    target_param.data.copy_(
                        source_param.data.to(
                            device=target_param.device,
                            dtype=target_param.dtype,
                        )
                    )
        main_print("***** GradLoRA calibration complete *****")

    if rank <= 0:
        wandb_tags = getattr(args, "wandb_tags", None)
        if isinstance(wandb_tags, str):
            wandb_tags = [tag.strip() for tag in wandb_tags.split(",") if tag.strip()]
        elif wandb_tags is not None:
            wandb_tags = [str(tag).strip() for tag in wandb_tags if str(tag).strip()]
        else:
            wandb_tags = None
        wandb_config = {
            key: value
            for key, value in vars(args).items()
            if not key.startswith("_")
        }
        wandb.init(
            project=args.project,
            config=wandb_config,
            name=args.run_name,
            tags=wandb_tags or None,
        )

    total_batch_size = world_size * args.gradient_accumulation_steps * args.train_batch_size
    main_print("***** Running SD3.5 training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size = {total_batch_size}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(f"  Master weight dtype: {next(transformer.parameters()).dtype}")

    progress_bar = tqdm(
        range(1, args.max_train_steps + 1),
        initial=init_steps,
        desc="Steps",
        disable=local_rank > 0,
    )
    step_times = deque(maxlen=100)

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for step in range(init_steps + 1, args.max_train_steps + 1):
            args.current_train_step = step
            start_time = time.time()
            total_loss = 0.0
            total_reward_pred_x0 = 0.0
            grad_norm = None
            traj_metrics = {}
            for inner_step in range(args.gradient_accumulation_steps):
                avg_loss, avg_reward_pred_x0, grad_norm, traj_metrics = train_sd35_one_step(
                    args,
                    inner_step,
                    device,
                    transformer,
                    vae,
                    fm_scheduler,
                    reward_model,
                    processor,
                    optimizer,
                    lr_scheduler,
                    loader,
                    preprocess_val,
                    select_idx_generator,
                )
                total_loss += avg_loss
                total_reward_pred_x0 += avg_reward_pred_x0

            if args.use_ema and ema_model is not None:
                for tgt, src in zip(ema_model.parameters(), transformer.parameters()):
                    tgt.data.lerp_(src.data.to(tgt), 1 - args.ema_decay)
            if step % args.checkpointing_steps == 0:
                if use_peft:
                    save_peft_checkpoint(
                        transformer,
                        rank,
                        args.output_dir,
                        step,
                        epoch,
                        adapter_type=adapter_type,
                    )
                else:
                    save_checkpoint(
                        transformer,
                        rank,
                        args.output_dir,
                        step,
                        epoch,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        save_optimizer_state=getattr(
                            args, "save_optimizer_state", False
                        ),
                    )
                if use_lora_pro and bool(args.lora_pro_save_optimizer_state):
                    _save_lorapro_optimizer_shard(
                        optimizer,
                        args.output_dir,
                        step,
                        epoch,
                        rank,
                    )
                if args.use_ema:
                    save_sharded_ema_checkpoint(ema_model, rank, args.output_dir, step, epoch)
                dist.barrier()

            step_time = time.time() - start_time
            step_times.append(step_time)
            progress_bar.set_postfix({"loss": f"{total_loss:.4f}", "step_time": f"{step_time:.2f}s", "grad_norm": grad_norm})
            if rank <= 0:
                log_dict = {
                    "train_loss": total_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": sum(step_times) / len(step_times),
                    "grad_norm": grad_norm,
                    "reward": total_reward_pred_x0,
                }
                log_dict.update({f"train/{name}": value for name, value in traj_metrics.items() if value is not None})
                wandb.log(log_dict, step=step)
            progress_bar.update(1)

            eval_interval = int(getattr(args, "evaluation_interval", 0) or 0)
            if eval_interval > 0 and (step % eval_interval == 0 or step == 1):
                transformer.eval()
                run_online_sd35_eval(
                    step,
                    args,
                    ema_model if args.use_ema else transformer,
                    vae,
                    rank,
                    world_size,
                    device,
                    reward_model if selected_reward == "hpsv3plusplus" else None,
                )
                transformer.train()

    if trainer_name == "flowbp_lagrange" and rank <= 0:
        csv_path = save_jk_sampling_csv(args.output_dir)
        if csv_path:
            print(f"[jk_sampling] Saved: {csv_path}")
    progress_bar.close()
    if dist.is_initialized():
        dist.barrier()
    if rank <= 0:
        wandb.finish()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
    elif dist.is_initialized():
        dist.destroy_process_group()


class LeapAlignSD35Trainer:
    def __init__(self, config):
        self.args = config

    def train(self):
        return run_training_sd35(self.args)
