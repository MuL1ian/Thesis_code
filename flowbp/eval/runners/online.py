from __future__ import annotations

import gc
import json
import os
import os.path as osp
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import wandb

from flowbp.eval.config import (
    EvalConfig,
    configure_hpsv3_env,
    configure_hpsv3plusplus_env,
)
from flowbp.eval.generation.flux import GeneratedEvalItem, sample_flux_images_distributed
from flowbp.eval.generation.sd35 import sample_sd35_images_distributed
from flowbp.eval.logging_wandb import build_wandb_eval_log
from flowbp.eval.metrics import as_float_list, collect_metric_values
from flowbp.eval.rewards.multi_scorer import build_reward_scorer
from flowbp.utils.logging_ import main_print


def _score_local_items(
    scorer: Any,
    local_items: list[GeneratedEvalItem],
) -> list[dict[str, Any]]:
    if not local_items:
        return []

    local_images = [item.image for item in local_items]
    local_prompts = [item.prompt for item in local_items]
    score_details, _ = scorer(local_images, local_prompts, metadata={})
    score_lists = {
        name: as_float_list(scores)
        for name, scores in score_details.items()
    }

    local_scores: list[dict[str, Any]] = []
    for item_idx, item in enumerate(local_items):
        score_item: dict[str, Any] = {
            "prompt_idx": item.prompt_idx,
            "img_idx": item.img_idx,
            "prompt": item.prompt,
        }
        for reward_name, reward_values in score_lists.items():
            if item_idx < len(reward_values):
                score_item[reward_name] = reward_values[item_idx]
        local_scores.append(score_item)
    return local_scores


def _append_hpsv3plusplus_scores(
    local_scores: list[dict[str, Any]],
    local_items: list[GeneratedEvalItem],
    hpsv3plusplus_scorer: Any,
) -> list[dict[str, Any]]:
    if not local_items:
        return local_scores

    if not local_scores:
        local_scores = [
            {
                "prompt_idx": item.prompt_idx,
                "img_idx": item.img_idx,
                "prompt": item.prompt,
            }
            for item in local_items
        ]

    local_images = [item.image for item in local_items]
    local_prompts = [item.prompt for item in local_items]
    rewards = hpsv3plusplus_scorer.reward(local_images, local_prompts)
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.as_tensor(rewards)
    rewards = rewards.detach().float()
    if rewards.ndim == 2 and rewards.shape[1] >= 1:
        rewards = rewards[:, 0]
    elif rewards.ndim > 1:
        rewards = rewards.reshape(rewards.shape[0], -1)[:, 0]
    values = as_float_list(rewards)
    for score_item, value in zip(local_scores, values):
        score_item["hpsv3plusplus"] = float(value)
    return local_scores


@torch.no_grad()
def run_online_flux_eval(
    step: int,
    args: Any,
    transformer,
    vae,
    rank: int,
    world_size: int,
    device: int | str | torch.device,
) -> None:
    """Run distributed FLUX eval and score outputs through eval reward fns."""

    eval_cfg = EvalConfig.from_args(args)
    main_print(f"--> Online Evaluation step {step}")

    hpsv3_paths = configure_hpsv3_env(eval_cfg)
    if hpsv3_paths:
        main_print(f"--> HPSv3 paths: {hpsv3_paths}")

    local_items = sample_flux_images_distributed(
        eval_cfg=eval_cfg,
        transformer=transformer,
        vae=vae,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    scorer, active_rewards = build_reward_scorer(
        reward_fn=eval_cfg.reward_fn,
        device=device,
        reward_ckpt_path=eval_cfg.reward_ckpt_path,
        allow_unavailable=True,
    )
    try:
        local_scores = _score_local_items(scorer, local_items)
    finally:
        scorer.to(torch.device("cpu"))
        del scorer

    all_scores_gathered = [None] * world_size
    dist.all_gather_object(all_scores_gathered, local_scores)

    if rank == 0:
        all_scores: list[dict[str, Any]] = []
        for rank_scores in all_scores_gathered:
            if rank_scores is not None:
                all_scores.extend(rank_scores)

        metric_names = list(active_rewards) + ["mean"]
        metric_values = collect_metric_values(all_scores, metric_names)
        metric_summary = " | ".join(
            f"{name}={float(np.mean(values)):.4f}"
            for name, values in metric_values.items()
        )
        num_eval_images = len(all_scores)
        main_print(
            f"--> Step {step} | Eval {metric_summary} | "
            f"num_images: {num_eval_images}"
        )

        wandb.log(
            build_wandb_eval_log(
                metric_values=metric_values,
                local_items=local_items,
                local_scores=local_scores,
                active_rewards=active_rewards,
                num_eval_images=num_eval_images,
            ),
            step=step,
        )

        if eval_cfg.output_dir:
            eval_log_dir = osp.join(eval_cfg.output_dir, "eval_logs")
            os.makedirs(eval_log_dir, exist_ok=True)
            eval_log_path = osp.join(eval_log_dir, f"eval_step_{step}.json")
            eval_log = {
                "step": step,
                "avg_scores": {
                    name: float(np.mean(values))
                    for name, values in metric_values.items()
                },
                "num_images": num_eval_images,
                "scores": all_scores,
            }
            with open(eval_log_path, "w", encoding="utf-8") as f:
                json.dump(eval_log, f, indent=2)

    torch.cuda.empty_cache()
    gc.collect()
    if world_size > 1:
        dist.barrier()


@torch.no_grad()
def run_online_sd35_eval(
    step: int,
    args: Any,
    transformer,
    vae,
    rank: int,
    world_size: int,
    device: int | str | torch.device,
    hpsv3plusplus_scorer: Any | None = None,
) -> None:
    """Run distributed SD3.5 eval and score outputs through eval reward fns."""

    eval_cfg = EvalConfig.from_args(args)
    main_print(f"--> Online SD3.5 Evaluation step {step}")

    hpsv3_paths = configure_hpsv3_env(eval_cfg)
    if hpsv3_paths:
        main_print(f"--> HPSv3 paths: {hpsv3_paths}")
    hpsv3plusplus_paths = configure_hpsv3plusplus_env(eval_cfg)
    if hpsv3plusplus_paths:
        main_print(f"--> HPSv3++ paths: {hpsv3plusplus_paths}")

    local_items = sample_sd35_images_distributed(
        eval_cfg=eval_cfg,
        transformer=transformer,
        vae=vae,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    requested_rewards = {
        str(name).lower(): float(weight)
        for name, weight in eval_cfg.reward_fn.items()
    }
    reuse_hpsv3plusplus = (
        "hpsv3plusplus" in requested_rewards and hpsv3plusplus_scorer is not None
    )
    scorer_reward_fn = dict(requested_rewards)
    if reuse_hpsv3plusplus:
        scorer_reward_fn.pop("hpsv3plusplus", None)

    scorer, active_rewards = build_reward_scorer(
        reward_fn=scorer_reward_fn,
        device=device,
        reward_ckpt_path=eval_cfg.reward_ckpt_path,
        allow_unavailable=True,
    )
    unavailable_rewards = getattr(scorer, "unavailable_rewards", {})
    if unavailable_rewards:
        main_print(f"--> SD3.5 eval unavailable rewards: {unavailable_rewards}")
    try:
        local_scores = _score_local_items(scorer, local_items)
    finally:
        scorer.to(torch.device("cpu"))
        del scorer
    if reuse_hpsv3plusplus:
        main_print("--> SD3.5 eval reusing training HPSv3++ reward model")
        local_scores = _append_hpsv3plusplus_scores(
            local_scores,
            local_items,
            hpsv3plusplus_scorer,
        )
        if "hpsv3plusplus" not in active_rewards:
            active_rewards = list(active_rewards) + ["hpsv3plusplus"]

    all_scores_gathered = [None] * world_size
    dist.all_gather_object(all_scores_gathered, local_scores)

    if rank == 0:
        all_scores: list[dict[str, Any]] = []
        for rank_scores in all_scores_gathered:
            if rank_scores is not None:
                all_scores.extend(rank_scores)

        metric_names = list(active_rewards) + ["mean"]
        metric_values = collect_metric_values(all_scores, metric_names)
        metric_summary = " | ".join(
            f"{name}={float(np.mean(values)):.4f}"
            for name, values in metric_values.items()
        )
        num_eval_images = len(all_scores)
        main_print(
            f"--> Step {step} | SD3.5 Eval {metric_summary} | "
            f"num_images: {num_eval_images}"
        )

        wandb.log(
            build_wandb_eval_log(
                metric_values=metric_values,
                local_items=local_items,
                local_scores=local_scores,
                active_rewards=active_rewards,
                num_eval_images=num_eval_images,
            ),
            step=step,
        )

        if eval_cfg.output_dir:
            eval_log_dir = osp.join(eval_cfg.output_dir, "eval_logs")
            os.makedirs(eval_log_dir, exist_ok=True)
            eval_log_path = osp.join(eval_log_dir, f"eval_sd35_step_{step}.json")
            eval_log = {
                "step": step,
                "avg_scores": {
                    name: float(np.mean(values))
                    for name, values in metric_values.items()
                },
                "num_images": num_eval_images,
                "scores": all_scores,
            }
            with open(eval_log_path, "w", encoding="utf-8") as f:
                json.dump(eval_log, f, indent=2)

    torch.cuda.empty_cache()
    gc.collect()
    if world_size > 1:
        dist.barrier()
