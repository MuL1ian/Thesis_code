"""Exact full-trajectory reward backpropagation for GradLoRA calibration."""

from __future__ import annotations

import torch

from flowbp.trainers.sd35.leapalign import (
    _model_forward_sd35,
    _prepare_timesteps,
    _sample_initial_latents,
)


def sample_fullbp_trajectory_sd35(
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
    """Roll out every flow step in one differentiable chain.

    This sampler is intentionally calibration-only. Unlike FlowBP-Sparse, it
    does not cache, detach, re-forward, or use a straight-through connector.
    The final reward gradient therefore traverses every Euler step.

    The Euler update mirrors ``_rollout_with_cache_sd35`` (explicit sigma
    indexing) so calibration does not depend on the mutable
    ``scheduler.step_index`` state used by ``scheduler.step()``.

    By default the CFG negative branch follows the training configuration.
    ``gradlora_fullbp_full_cfg_grad`` can enable exact gradients through both
    CFG branches, at substantially higher activation-memory cost.
    """
    del generator
    args._last_sample_keep_mask = None
    args._last_connector_payload = None

    latents_xt, _, _ = _sample_initial_latents(
        args, transformer, device, prompt_embeds.shape[0]
    )
    timesteps = _prepare_timesteps(
        args, fm_scheduler, transformer, latents_xt, device
    )
    total_steps = int(timesteps.size(0))
    if total_steps < 1:
        raise ValueError("FullBP calibration requires at least one rollout step")
    if len(fm_scheduler.sigmas) < total_steps + 1:
        raise RuntimeError(
            "FullBP calibration requires scheduler.sigmas to cover every "
            f"rollout step; got {len(fm_scheduler.sigmas)} sigmas for "
            f"{total_steps} steps"
        )

    latents_pred_x0 = None
    full_cfg_grad = bool(
        getattr(args, "gradlora_fullbp_full_cfg_grad", False)
        or getattr(args, "loraga_fullbp_full_cfg_grad", False)
    )
    for idx, step_t in enumerate(timesteps):
        pred = _model_forward_sd35(
            args,
            transformer,
            latents_xt,
            step_t,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            detach_negative_branch_override=False if full_cfg_grad else None,
        )
        sigma = fm_scheduler.sigmas[idx].to(
            device=latents_xt.device, dtype=torch.float32
        )
        sigma_next = fm_scheduler.sigmas[idx + 1].to(
            device=latents_xt.device, dtype=torch.float32
        )
        latents_f32 = latents_xt.float()
        pred_f32 = pred.float()
        if idx == total_steps - 1:
            # Equivalent to the final Euler state when sigma_next == 0, and to
            # FlowMatchEulerDiscreteWithPredX0Scheduler(..., with_pred_x0=True).
            latents_pred_x0 = latents_f32 + (-sigma) * pred_f32
            latents_xt = latents_pred_x0.to(dtype=pred.dtype)
        else:
            latents_xt = (latents_f32 + (sigma_next - sigma) * pred_f32).to(
                dtype=pred.dtype
            )

    if latents_pred_x0 is None:
        raise RuntimeError("FullBP rollout did not produce a final x0 prediction")
    if latents_pred_x0.dtype != torch.bfloat16:
        latents_pred_x0 = latents_pred_x0.to(torch.bfloat16)

    sample_weights = torch.ones(
        latents_pred_x0.shape[0],
        device=device,
        dtype=torch.float32,
    )
    traj_metrics = {
        "fullbp_steps": torch.tensor(
            float(total_steps), device=device, dtype=torch.float32
        ),
        "d0": None,
        "dj": None,
        "w_sim": None,
    }
    return latents_pred_x0, sample_weights, traj_metrics
