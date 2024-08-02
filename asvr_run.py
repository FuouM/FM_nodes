
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from comfy.utils import ProgressBar

from .constants import ASVR_MODEL_PATH
from .asvr_module.asvr_model import ASVR_Config, ASVR

base_dir = Path(__file__).resolve().parent

class ASVR_VideoSR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_video": ("IMAGE",),
                "height_scale": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.1, "step": 0.01},
                ),
                "width_scale": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.1, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_video",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes"

    def todo(
        self,
        src_video: torch.Tensor,
        height_scale: float,
        width_scale: float
    ):
        asvr_runner = ASVR(ASVR_Config(
            f"{base_dir}/{ASVR_MODEL_PATH}",
            space_scale=(height_scale, width_scale)
        ))
        output = asvr_runner.inference(src_video)
        return (output, )
