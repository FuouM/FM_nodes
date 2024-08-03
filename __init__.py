from .run import (
    WFEN, RealViFormerSR, ProPIH_Harmonizer, CoLIE_LowLight_Enhance, PCSR_4X
)

NODE_CLASS_MAPPINGS = {
    "WFEN": WFEN,
    "RealViFormerSR": RealViFormerSR,
    "ProPIH_Harmonizer": ProPIH_Harmonizer,
    "CoLIE_LowLight_Enhance": CoLIE_LowLight_Enhance,
    "PCSR_4X": PCSR_4X,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WFEN": "WFEN Face Restore",
    "RealViFormerSR": "RealViFormer Video SR",
    "ProPIH_Harmonizer": "ProPIH Harmonizer",
    "CoLIE_LowLight_Enhance": "CoLIE LowLight Enhance",
    "PCSR_4X": "PCSR 4X",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]