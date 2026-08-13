"""Entrypoint for FlowBP training on FLUX.2.

Parallels :mod:`flowbp.train_flowbp_flux` but loads the Flux2 trainer
classes. Shares the same YAML config schema; FLUX.2-specific knobs live under
the optional ``flux2:`` section.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowbp.config.parser import build_parser as _build_parser
from flowbp.config.parser import parse_args as _parse_args

BACKEND = "flux2"


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(BACKEND)


def parse_args() -> argparse.Namespace:
    return _parse_args(BACKEND)


def main() -> None:
    args = parse_args()
    # Import FLUX.2 trainers directly so dependency errors surface with a
    # clear stack trace.
    from flowbp.trainers.flux2.leapalign import LeapAlignFlux2Trainer
    from flowbp.trainers.flux2.flowbp_lagrange import (
        FlowBPLagrangeFlux2Trainer,
    )
    from flowbp.trainers.flux2.refl import LeapAlignReFLFlux2Trainer
    from flowbp.trainers.flux2.flowbp_sparse import (
        FlowBPSparseFlux2Trainer,
    )
    from flowbp.trainers.flux2.flowbp_bridge import (
        FlowBPBridgeFlux2Trainer,
    )
    from flowbp.trainers.flux2.drtune import LeapAlignDRTuneFlux2Trainer
    from flowbp.trainers.flux2.draft_lv import LeapAlignDRaFTLVFlux2Trainer

    trainer_map = {
        "leapalign": LeapAlignFlux2Trainer,
        "flowbp_lagrange": FlowBPLagrangeFlux2Trainer,
        "refl": LeapAlignReFLFlux2Trainer,
        "flowbp_sparse": FlowBPSparseFlux2Trainer,
        "flowbp_bridge": FlowBPBridgeFlux2Trainer,
        "drtune": LeapAlignDRTuneFlux2Trainer,
        "draft_lv": LeapAlignDRaFTLVFlux2Trainer,
    }
    if args.trainer not in trainer_map:
        raise ValueError(
            "FLUX.2 entrypoint supports flowbp_sparse/flowbp_bridge/"
            "flowbp_lagrange and baselines leapalign/refl/draft_lv/drtune, "
            f"got {args.trainer!r}."
        )
    trainer_map[args.trainer](args).train()


if __name__ == "__main__":
    main()
