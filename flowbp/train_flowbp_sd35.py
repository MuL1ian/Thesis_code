"""FlowBP SD3.5 fine-tuning entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowbp.config.parser import build_parser as _build_parser
from flowbp.config.parser import parse_args as _parse_args

BACKEND = "sd35"


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(BACKEND)


def parse_args() -> argparse.Namespace:
    return _parse_args(BACKEND)


def main() -> None:
    args = parse_args()
    from flowbp.trainers.sd35 import (
        FlowBPBridgeSD35Trainer,
        FlowBPLagrangeSD35Trainer,
        FlowBPSparseSD35Trainer,
        LeapAlignDRTuneSD35Trainer,
        LeapAlignDRaFTLVSD35Trainer,
        LeapAlignReFLSD35Trainer,
        LeapAlignSD35Trainer,
    )

    trainer_map = {
        # FullBP is exposed for memory/throughput benchmarking. Its sampler
        # differentiates through the complete rollout and is usually too
        # memory-intensive for regular training.
        "fullbp": LeapAlignSD35Trainer,
        "leapalign": LeapAlignSD35Trainer,
        "refl": LeapAlignReFLSD35Trainer,
        "flowbp_lagrange": FlowBPLagrangeSD35Trainer,
        "flowbp_sparse": FlowBPSparseSD35Trainer,
        "flowbp_bridge": FlowBPBridgeSD35Trainer,
        "draft_lv": LeapAlignDRaFTLVSD35Trainer,
        "drtune": LeapAlignDRTuneSD35Trainer,
    }
    if args.trainer not in trainer_map:
        raise ValueError(
            "SD3.5 entrypoint supports fullbp/flowbp_sparse/flowbp_bridge/"
            "flowbp_lagrange and baselines leapalign/refl/draft_lv/drtune, "
            f"got {args.trainer!r}."
        )
    trainer_map[args.trainer](args).train()


if __name__ == "__main__":
    main()
