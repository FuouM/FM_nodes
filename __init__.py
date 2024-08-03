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
    from .vfi_mamba_run import VFI_MAMBA
    NODE_CLASS_MAPPINGS["VFI_MAMBA"] = VFI_MAMBA
    NODE_DISPLAY_NAME_MAPPINGS["VFI_MAMBA"] = "VFI Mamba"
except ImportError as e:
    print(f"Failed to load VFI_MAMBA. This will not affect other nodes. {e}")
    pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]