from pathlib import Path

import torch

from .constants import VFI_MAMBA_DEFAULT, VFI_MAMBA_MODELS, VFI_MAMBA_PATHS
from .vfimamba_module.vfi_mamba_model import run_vfi_mamba

base_dir = Path(__file__).resolve().parent


class VFI_MAMBA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_video": ("IMAGE",),
                "vfi_scale": (
                    "INT",
                    {"default": 2, "min": 2, "step": 1},
                ),
                "model": (VFI_MAMBA_MODELS, {"default": VFI_MAMBA_DEFAULT}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_video",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes"

    def todo(
        self, src_video: torch.Tensor, vfi_scale: int, model: str
    ):
        return (
            run_vfi_mamba(
                src_video=src_video,
                model_name=model,
                model_path=f"{base_dir}/{VFI_MAMBA_PATHS[model]}",
                vfi_scale=vfi_scale,
                scale=0,
            ),
        )
