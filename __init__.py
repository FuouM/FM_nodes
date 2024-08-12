from .run import (
    WFEN,
    CoLIE_LowLight_Enhance,
    ProPIH_Harmonizer,
    RealViFormerSR,
)
from .run_convir import (
    ConvIR_DefocusDeblur,
    ConvIR_DeHaze,
    ConvIR_DeRain,
    ConvIR_DeSnow,
    ConvIR_MotionDeBlur,
)
from .run_stabstitch import (
    StabStitch_Crop_Resize,
    StabStitch_Stabilize,
    StabStitch_Stitch,
)

NODE_CLASS_MAPPINGS = {
    "WFEN": WFEN,
    "RealViFormerSR": RealViFormerSR,
    "ProPIH_Harmonizer": ProPIH_Harmonizer,
    "CoLIE_LowLight_Enhance": CoLIE_LowLight_Enhance,
    "ConvIR_DeHaze": ConvIR_DeHaze,
    "ConvIR_DeRain": ConvIR_DeRain,
    "ConvIR_DeSnow": ConvIR_DeSnow,
    "ConvIR_MotionDeBlur": ConvIR_MotionDeBlur,
    "ConvIR_DefocusDeblur": ConvIR_DefocusDeblur,
    "StabStitch": StabStitch_Stitch,
    "StabStitch_Stabilize": StabStitch_Stabilize,
    "StabStitch_Crop_Resize": StabStitch_Crop_Resize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WFEN": "WFEN Face Restore",
    "RealViFormerSR": "RealViFormer Video SR",
    "ProPIH_Harmonizer": "ProPIH Harmonizer",
    "CoLIE_LowLight_Enhance": "CoLIE LowLight Enhance",
    "ConvIR_DeHaze": "ConvIR DeHaze",
    "ConvIR_DeRain": "ConvIR DeRain",
    "ConvIR_DeSnow": "ConvIR DeSnow",
    "ConvIR_MotionDeBlur": "ConvIR Motion DeBlur",
    "ConvIR_DefocusDeblur": "ConvIR Defocus Deblur",
    "StabStitch": "StabStitch",
    "StabStitch_Stabilize": "StabStitch Stabilize",
    "StabStitch_Crop_Resize": "StabStitch Crop Resize",
}

try:
    from .vfi_mamba_run import VFI_MAMBA

    NODE_CLASS_MAPPINGS["VFI_MAMBA"] = VFI_MAMBA
    NODE_DISPLAY_NAME_MAPPINGS["VFI_MAMBA"] = "VFI Mamba"
except ImportError as e:
    print(f"Failed to load VFI_MAMBA. This will not affect other nodes. {e}")
    pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
