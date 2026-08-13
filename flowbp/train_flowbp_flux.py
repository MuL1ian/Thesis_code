"""FlowBP FLUX.1 fine-tuning entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowbp.config.parser import build_parser as _build_parser
from flowbp.config.parser import parse_args as _parse_args

BACKEND = "flux1"


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(BACKEND)


def parse_args() -> argparse.Namespace:
    return _parse_args(BACKEND)


def main() -> None:
    args = parse_args()
    from flowbp.trainers import (
        FlowBPSparseFluxTrainer,
        FlowBPBridgeFluxTrainer,
        LeapAlignDRaFTLVFluxTrainer,
        LeapAlignDRTuneFluxTrainer,
        LeapAlignFluxTrainer,
        FlowBPLagrangeFluxTrainer,
        LeapAlignReFLFluxTrainer,
    )

    trainer_map = {
        "leapalign": LeapAlignFluxTrainer,
        "refl": LeapAlignReFLFluxTrainer,
        "flowbp_lagrange": FlowBPLagrangeFluxTrainer,
        "flowbp_sparse": FlowBPSparseFluxTrainer,
        "flowbp_bridge": FlowBPBridgeFluxTrainer,
        "drtune": LeapAlignDRTuneFluxTrainer,
        "draft_lv": LeapAlignDRaFTLVFluxTrainer,
    }
    if args.trainer not in trainer_map:
        raise ValueError(
            "FLUX.1 entrypoint supports flowbp_sparse/flowbp_bridge/"
            "flowbp_lagrange and baselines leapalign/refl/draft_lv/drtune, "
            f"got {args.trainer!r}."
        )
    trainer_map[args.trainer](args).train()


if __name__ == "__main__":
    main()
