from .run import WFEN, RealViFormerSR, ProPIH_Harmonizer, CoLIE_LowLight_Enhance

from .run_convir import (
    ConvIR_DeHaze,
    ConvIR_DeRain,
    ConvIR_DeSnow,
    ConvIR_MotionDeBlur,
    ConvIR_DefocusDeblur,
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
}

try:
    from .vfi_mamba_run import VFI_MAMBA

    NODE_CLASS_MAPPINGS["VFI_MAMBA"] = VFI_MAMBA
    NODE_DISPLAY_NAME_MAPPINGS["VFI_MAMBA"] = "VFI Mamba"
except ImportError as e:
    print(f"Failed to load VFI_MAMBA. This will not affect other nodes. {e}")
    pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
