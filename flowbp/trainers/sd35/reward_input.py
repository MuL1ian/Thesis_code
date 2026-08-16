"""Reward-model input axis for the SD3.5 direct-gradient baselines.

ReFL and DRTune stop the rollout early and score the per-step posterior mean,
while the FlowBP variants score the sampled endpoint. These helpers let a
baseline keep its active set, integration weights and optimizer untouched and
swap only what the reward model sees, which is what isolates the reward-input
axis in the ablation table.

The rollout continuation runs under ``no_grad``, so the extra denoising steps
cost time but no activation memory.
"""

from __future__ import annotations

import torch

from flowbp.trainers.sd35.leapalign import _model_forward_sd35

SUPPORTED_REWARD_INPUT_MODES = ("posterior_mean", "sampled_x0")


def reward_input_mode(args) -> str:
    """Return the validated reward-input mode for ``args``."""
    mode = str(
        getattr(args, "reward_input_mode", "posterior_mean") or "posterior_mean"
    ).strip().lower()
    if mode not in SUPPORTED_REWARD_INPUT_MODES:
        raise ValueError(
            f"Unsupported reward_input_mode={mode!r}; expected one of "
            f"{SUPPORTED_REWARD_INPUT_MODES}."
        )
    return mode


def uses_sampled_endpoint(args) -> bool:
    return reward_input_mode(args) == "sampled_x0"


@torch.no_grad()
def finish_rollout_sd35(
    args,
    transformer,
    fm_scheduler,
    latents,
    timesteps,
    start_idx,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
):
    """Run the remaining denoising steps and return the sampled endpoint.

    ``latents`` must be the state produced by the caller's last
    ``fm_scheduler.step``, so the scheduler's internal step index already
    points at ``start_idx``. An empty range means the caller already reached
    the endpoint, in which case ``latents`` is returned unchanged.
    """
    latents = latents.detach()
    if latents.dtype != torch.bfloat16:
        latents = latents.to(torch.bfloat16)

    for idx in range(int(start_idx), timesteps.size(0)):
        step_t = timesteps[idx]
        pred = _model_forward_sd35(
            args,
            transformer,
            latents,
            step_t,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        latents = fm_scheduler.step(
            pred,
            step_t,
            latents,
            with_pred_x0=False,
        )[0]
        if latents.dtype != torch.bfloat16:
            latents = latents.to(torch.bfloat16)

    return latents.detach()


def attach_sampled_endpoint(pred_x0, x0_true):
    """Straight-through swap of the reward input.

    The forward value becomes the sampled ``x0_true`` that the reward model
    scores, while the backward path still flows through ``pred_x0`` and
    therefore through whatever active set the caller built.
    """
    return pred_x0 + (x0_true.to(pred_x0.dtype) - pred_x0).detach()
