"""
@author: Fuou Marinas
@title: FM Nodes
@nickname: FM_nodes
@description: A collection of nodes. WFEN Face Super Resolution.
"""

from pathlib import Path

import torch

from .constants import WFEN_MODEL_PATH
from .utils import ensure_size
from .wfen_module.wfen_model import WFENModel

base_dir = Path(__file__).resolve().parent


class WFEN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_img",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes"

    def todo(
        self,
        src_img: torch.Tensor,
    ):
        print(f"{src_img.shape=}")
        source_tensor = ensure_size(src_img.permute(0, 3, 1, 2))
        print(f"{source_tensor.shape=}")
        wfening = WFENModel()
        wfening.for_load_pretrain_model(f"{base_dir}/{WFEN_MODEL_PATH}")

        out_tensor = wfening.netG.forward(source_tensor)
        # out_tensor = ensure_size(out_tensor, size=src_img.shape[1:3])
        print(f"{out_tensor.shape=}")
        out_tensor = out_tensor.permute(0, 2, 3, 1)

        return (out_tensor,)
