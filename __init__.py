from .run import (
    WFEN, RealViFormerSR, ProPIH_Harmonizer
)

NODE_CLASS_MAPPINGS = {
    "WFEN": WFEN,
    "RealViFormerSR": RealViFormerSR,
    "ProPIH_Harmonizer": ProPIH_Harmonizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WFEN": "WFEN Face Restore",
    "RealViFormerSR": "RealViFormer Video SR",
    "ProPIH_Harmonizer": "ProPIH Harmonizer",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]