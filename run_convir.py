from pathlib import Path

import torch
from comfy.utils import ProgressBar

from .module_convir.constants_convir import (
    DEBLUR_DEFAULT,
    DEBLUR_MODELS,
    DEFOCUS_MODEL,
    DEFOCUS_SIZES,
    DEHAZE_DEFAULT,
    DEHAZE_MODELS,
    DEHAZE_SIZES,
    DERAIN_MODEL,
    DERAIN_SIZE,
    DESNOW_DEFAULT,
    DESNOW_MODELS,
    DESNOW_SIZES,
    MODEL_DIR,
    SIZE_BASE,
    SIZE_LARGE,
    get_deblur_model,
    get_defocus_model,
    get_dehaze_model,
    get_derain_model,
    get_desnow_model,
)
from .module_convir.ConvIR import ConvIR
from .module_convir.convir_utils import preprocess_image

base_dir = Path(__file__).resolve().parent


def run_convir(src_img, model_name, model_size, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_img, h, w = preprocess_image(src_img, device)

    model = ConvIR(version=model_size, data=model_name)

    checkpoint_path = f"{base_dir}/{MODEL_DIR}/{model_file}"
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict["model"])
    model = model.to(device)
    model.eval()

    num_frames = src_img.size(0)
    pbar = ProgressBar(num_frames)
    result: list[torch.Tensor] = []

    for i in range(num_frames):
        image = src_img[i].unsqueeze(0)
        pred = model.forward(image)[2]
        pred = pred[:, :, :h, :w]
        pred_clip = torch.clamp(pred, 0, 1)
        result.append(pred_clip)
        pbar.update_absolute(i, num_frames)

    result_ts = torch.cat(result, dim=0).permute(0, 2, 3, 1)
    return result_ts


class ConvIR_DeHaze:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "model_name": (DEHAZE_MODELS, {"default": DEHAZE_DEFAULT}),
                "model_size": (DEHAZE_SIZES, {"default": SIZE_BASE}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_img",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes/ConvIR"

    def todo(self, src_img: torch.Tensor, model_name: str, model_size: str):
        model_file, model_size = get_dehaze_model(model_name, model_size)

        result_ts = run_convir(src_img, model_name, model_size, model_file)
        return (result_ts,)


class ConvIR_DeRain:
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
    CATEGORY = "FM_nodes/ConvIR"

    def todo(self, src_img: torch.Tensor):
        model_file = get_derain_model()

        result_ts = run_convir(src_img, DERAIN_MODEL, DERAIN_SIZE, model_file)
        return (result_ts,)


class ConvIR_DeSnow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "model_name": (DESNOW_MODELS, {"default": DESNOW_DEFAULT}),
                "model_size": (DESNOW_SIZES, {"default": SIZE_BASE}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_img",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes/ConvIR"

    def todo(self, src_img: torch.Tensor, model_name: str, model_size: str):
        model_file = get_desnow_model(model_name, model_size)

        result_ts = run_convir(src_img, model_name, model_size, model_file)
        return (result_ts,)


class ConvIR_MotionDeBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "model_name": (DEBLUR_MODELS, {"default": DEBLUR_DEFAULT}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_img",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes/ConvIR"

    def todo(self, src_img: torch.Tensor, model_name: str):
        model_file = get_deblur_model(model_name)

        result_ts = run_convir(src_img, model_name, SIZE_LARGE, model_file)
        return (result_ts,)


class ConvIR_DefocusDeblur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "model_size": (DEFOCUS_SIZES, {"default": SIZE_BASE}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_img",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes/ConvIR"

    def todo(self, src_img: torch.Tensor, model_size: str):
        model_file = get_defocus_model(model_size)

        result_ts = run_convir(src_img, DEFOCUS_MODEL, model_size, model_file)
        return (result_ts,)
