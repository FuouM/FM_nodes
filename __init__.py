from .run import (
    WFEN, RealViFormerSR,
)

NODE_CLASS_MAPPINGS = {
    "WFEN": WFEN,
    "RealViFormerSR": RealViFormerSR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealViFormerSR": "RealViFormer Video SR"
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]