from .run import (
    WFEN, RealViFormerSR, ProPIH_Harmonizer, CoLIE_LowLight_Enhance
)

NODE_CLASS_MAPPINGS = {
    "WFEN": WFEN,
    "RealViFormerSR": RealViFormerSR,
    "ProPIH_Harmonizer": ProPIH_Harmonizer,
    "CoLIE_LowLight_Enhance": CoLIE_LowLight_Enhance,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "WFEN": "WFEN Face Restore",
    "RealViFormerSR": "RealViFormer Video SR",
    "ProPIH_Harmonizer": "ProPIH Harmonizer",
    "CoLIE_LowLight_Enhance": "CoLIE LowLight Enhance",
}

try:
    from .asvr_run import ASVR_VideoSR
    NODE_CLASS_MAPPINGS["ASVR_VideoSR"] = ASVR_VideoSR
    NODE_DISPLAY_NAME_MAPPINGS["ASVR_VideoSR"] = "ASVR Video SR"
except ImportError as e:
    print(f"Failed to load ASVR_VideoSR. This will not affect other nodes. {e}")
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]