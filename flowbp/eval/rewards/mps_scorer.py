"""Offline evaluator for the overall-preference MPS checkpoint."""

from __future__ import annotations

import torch
from torch import nn

from flowbp.eval.rewards import reward_ckpt_path
from flowbp.rewards.mps_reward import MPSDifferentiableReward


class MPSScorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        del dtype
        self.device = torch.device(device)
        self.reward = MPSDifferentiableReward(
            device=self.device,
            checkpoint_path=reward_ckpt_path.CKPT_PATH,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        target = kwargs.get("device")
        if target is None and args and isinstance(args[0], (str, int, torch.device)):
            target = args[0]
        if target is not None:
            self.device = (
                torch.device("cuda", target)
                if isinstance(target, int)
                else torch.device(target)
            )
            self.reward.device = self.device
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor, prompts) -> torch.Tensor:
        images = images.to(self.device, dtype=torch.float32, non_blocking=True)
        return self.reward(images, prompts).detach().float().contiguous()
