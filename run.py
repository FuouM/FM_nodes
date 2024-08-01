"""
@author: Fuou Marinas
@title: FM Nodes
@nickname: FM_nodes
@description: A collection of nodes. WFEN Face Super Resolution.
"""

from pathlib import Path
import tqdm

import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar

from .propih_module.propih_model import VGG19HRNetModel

from .constants import (
    PROPIH_G_MODEL_PATH,
    PROPIH_VGG_MODEL_PATH,
    REALVIFORMER_MODEL_PATH,
    WFEN_MODEL_PATH,
)
from .realviformer_module.realviformer_arch import RealViformer
from .utils import ensure_size, img_to_mask
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
        print(f"{out_tensor.shape=}")
        out_tensor = out_tensor.permute(0, 2, 3, 1)

        return (out_tensor,)


def inference_realviformer(images: torch.Tensor, model: RealViformer) -> torch.Tensor:
    padded = False
    with torch.no_grad():
        h, w = images.shape[-2:]
        ah, aw = h % 4, w % 4
        padh = 0 if ah == 0 else 4 - ah
        padw = 0 if aw == 0 else 4 - aw
        if padh != 0 or padw != 0:
            padded = True
            images = F.pad(
                images.squeeze(0), pad=(padw, 0, padh, 0), mode="reflect"
            ).unsqueeze(0)
        outputs = model(images)

    if padded:
        outputs = outputs[..., padh * 4 :, padw * 4 :]

    return outputs


class RealViFormerSR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_video": ("IMAGE",),
                "interval": (
                    "INT",
                    {"default": 50, "min": 0, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_video",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes"

    def todo(self, src_video: torch.Tensor, interval: int):
        src_video = src_video.permute(0, 3, 1, 2)
        print(f"{src_video.shape=}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RealViformer(
            num_feat=48,
            num_blocks=[2, 3, 4, 1],
            spynet_path=None,
            heads=[1, 2, 4],
            ffn_expansion_factor=2.66,
            merge_head=2,
            bias=False,
            LayerNorm_type="BiasFree",
            ch_compress=True,
            squeeze_factor=[4, 4, 4],
            masked=True,
        )
        model.load_state_dict(
            torch.load(f"{base_dir}/{REALVIFORMER_MODEL_PATH}")["params"], strict=False
        )
        model.eval()
        model = model.to(device)

        if src_video.shape[0] <= interval:
            out_tensor = inference_realviformer(
                src_video.unsqueeze(0).to(device), model
            )
            out_tensor = out_tensor.squeeze(dim=0).permute(0, 2, 3, 1)
            return (out_tensor,)

        num_imgs = src_video.shape[0]
        outputs: list[torch.Tensor] = []
        pbar = ProgressBar(num_imgs)

        for idx in tqdm.tqdm(range(0, num_imgs, interval)):
            interval = min(interval, num_imgs - idx)
            imgs = src_video[idx : idx + interval]
            imgs = imgs.unsqueeze(0).to(device)  # [b, n, c, h, w]
            outputs.append(inference_realviformer(imgs, model).squeeze(dim=0))
            pbar.update_absolute(idx + interval, num_imgs)

        out_tensor = torch.cat(outputs, dim=0).permute(0, 2, 3, 1)
        return (out_tensor,)


class ProPIH_Harmonizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "composite": ("IMAGE",),
                "background": ("IMAGE",),
            },
            "optional": {"foreground_mask": ("IMAGE",), "foreground_MASK": ("MASK",)},
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "out_0",
        "out_1",
        "out_2",
        "out_3",
    )
    FUNCTION = "todo"
    CATEGORY = "FM_nodes"

    def todo(
        self,
        composite: torch.Tensor,
        background: torch.Tensor,
        foreground_mask: torch.Tensor | None = None,
        foreground_MASK: torch.Tensor | None = None,
    ):
        if foreground_mask is None and foreground_MASK is None:
            raise ValueError("Please provide one mask image")

        if foreground_MASK is not None:
            mask = foreground_MASK.unsqueeze(dim=0)
        else:
            mask = img_to_mask(foreground_mask.permute(0, 3, 1, 2))

        composite = composite.permute(0, 3, 1, 2)
        background = background.permute(0, 3, 1, 2)

        propih = VGG19HRNetModel(
            vgg_path=f"{base_dir}/{PROPIH_VGG_MODEL_PATH}",
            g_path=f"{base_dir}/{PROPIH_G_MODEL_PATH}",
        )
        outputs = propih.forward(comp=composite, style=background, mask=mask)
        final_outputs = []
        for ts in outputs:
            final_outputs.append(ts.permute(0, 2, 3, 1))

        return (
            final_outputs[0],
            final_outputs[1],
            final_outputs[2],
            final_outputs[3],
        )
