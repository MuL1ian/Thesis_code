"""Bridge coupling coefficient applied to the anchor's model input."""

from __future__ import annotations

import torch


def alpha_coupled_input(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Scale the gradient flowing through ``x`` by ``alpha``, leaving its value intact.

    The convex combination is the identity on the forward value, so the sampled
    trajectory -- and therefore the reward cotangent at ``x0`` -- does not depend
    on ``alpha`` at all. That is what makes the parameter gradient exactly affine
    in ``alpha``, and it is the assumption every alpha diagnostic rests on.

    The identity is an fp32 property. bf16 carries 8 significand bits, so
    ``0.3 * x + 0.7 * x`` lands one to two ulp away from ``x`` and only
    ``alpha in {0, 0.5, 1}`` survives exactly. Mixing in bf16 would therefore let
    the deployed alpha perturb the trajectory by a rounding artefact of the same
    order as the coupling effect being measured, so mix in fp32 and cast back.
    """
    alpha = float(alpha)
    if alpha == 1.0:
        return x
    if alpha == 0.0:
        return x.detach()
    x32 = x.float()
    return (alpha * x32 + (1.0 - alpha) * x32.detach()).to(x.dtype)
