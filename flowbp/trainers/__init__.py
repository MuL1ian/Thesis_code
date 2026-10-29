# SD3.5 trainers require diffusers SD3 components. Keep this optional so the
# config and reward modules can still be imported without them.
try:
    from flowbp.trainers.sd35 import (
        FlowBPBridgeSD35Trainer,
        FlowBPLagrangeSD35Trainer,
        FlowBPSparseSD35Trainer,
        LeapAlignDRTuneSD35Trainer,
        LeapAlignDRaFTLVSD35Trainer,
        LeapAlignReFLSD35Trainer,
        LeapAlignSD35Trainer,
    )

    _SD35_AVAILABLE = True
except ImportError:
    FlowBPBridgeSD35Trainer = None
    FlowBPLagrangeSD35Trainer = None
    FlowBPSparseSD35Trainer = None
    LeapAlignDRTuneSD35Trainer = None
    LeapAlignDRaFTLVSD35Trainer = None
    LeapAlignReFLSD35Trainer = None
    LeapAlignSD35Trainer = None
    _SD35_AVAILABLE = False


__all__ = [
    "LeapAlignSD35Trainer",
    "LeapAlignReFLSD35Trainer",
    "LeapAlignDRaFTLVSD35Trainer",
    "LeapAlignDRTuneSD35Trainer",
    "FlowBPSparseSD35Trainer",
    "FlowBPBridgeSD35Trainer",
    "FlowBPLagrangeSD35Trainer",
]
