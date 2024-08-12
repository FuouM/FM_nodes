from pathlib import Path

import torch

from .constants import (
    STAB_CONSTRUCT_DEFAULT,
    STAB_CONSTRUCT_MODES,
    STAB_SMOOTH_PATH,
    STAB_SPATIAL_PATH,
    STAB_TEMPORAL_PATH,
)
from .module_stabstitch.stabstitch_model import (
    crop_or_resize,
    stabstitch_inference,
    stabstitch_single_inference,
)

base_dir = Path(__file__).resolve().parent


class StabStitch_Stitch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_1": ("IMAGE",),
                "video_2": ("IMAGE",),
                # mode = 'FAST': use F.grid_sample to interpolate. It's fast, but may produce thin black boundary.
                # mode = 'NORMAL': use our implemented interpolation function. It's a bit slower, but avoid the black boundary.
                "interpolate_mode": (
                    STAB_CONSTRUCT_MODES,
                    {"default": STAB_CONSTRUCT_DEFAULT},
                ),
                "do_linear_blend": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "combined",
        "stab",
        "mask",
    )
    FUNCTION = "todo"
    CATEGORY = "FM_nodes/StabStitch"

    def todo(
        self,
        video_1: torch.Tensor,
        video_2: torch.Tensor,
        interpolate_mode: str,
        do_linear_blend: bool,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stable_list, video_frames_2, mask_frames_2 = stabstitch_inference(
            video_1,
            video_2,
            spatial_path=f"{base_dir}/{STAB_SPATIAL_PATH}",
            temporal_path=f"{base_dir}/{STAB_TEMPORAL_PATH}",
            smooth_path=f"{base_dir}/{STAB_SMOOTH_PATH}",
            device=device,
            mode=interpolate_mode,
            do_linear_blend=do_linear_blend,
        )

        out_video = torch.cat(stable_list, dim=0).permute(0, 2, 3, 1)
        out_mask = torch.cat(mask_frames_2, dim=0).permute(0, 2, 3, 1)
        out_img2 = torch.cat(video_frames_2, dim=0)[:, [2, 1, 0], :, :].permute(
            0, 2, 3, 1
        )

        return (
            out_video,
            out_img2,
            out_mask,
        )


class StabStitch_Stabilize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_1": ("IMAGE",),
                # mode = 'FAST': use F.grid_sample to interpolate. It's fast, but may produce thin black boundary.
                # mode = 'NORMAL': use our implemented interpolation function. It's a bit slower, but avoid the black boundary.
                "interpolate_mode": (
                    STAB_CONSTRUCT_MODES,
                    {"default": STAB_CONSTRUCT_DEFAULT},
                ),
                "do_linear_blend": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "combined",
        "stab",
        "mask",
    )
    FUNCTION = "todo"
    CATEGORY = "FM_nodes/StabStitch"

    def todo(
        self,
        video_1: torch.Tensor,
        interpolate_mode: str,
        do_linear_blend: bool,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stable_list, video_frames_2, mask_frames_2 = stabstitch_single_inference(
            video_1,
            spatial_path=f"{base_dir}/{STAB_SPATIAL_PATH}",
            temporal_path=f"{base_dir}/{STAB_TEMPORAL_PATH}",
            smooth_path=f"{base_dir}/{STAB_SMOOTH_PATH}",
            device=device,
            mode=interpolate_mode,
            do_linear_blend=do_linear_blend,
        )

        out_video = torch.cat(stable_list, dim=0).permute(0, 2, 3, 1)
        out_mask = torch.cat(mask_frames_2, dim=0).permute(0, 2, 3, 1)
        out_img2 = torch.cat(video_frames_2, dim=0)[:, [2, 1, 0], :, :].permute(
            0, 2, 3, 1
        )

        return (
            out_video,
            out_img2,
            out_mask,
        )


class StabStitch_Crop_Resize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_1": ("IMAGE",),
                "crop_side": (
                    ["left", "right"],
                    {"default": "left"},
                ),
                "target_h": (
                    "INT",
                    {"default": 360, "min": 1, "step": 1},
                ),
                "target_w": (
                    "INT",
                    {"default": 480, "min": 1, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized",)
    FUNCTION = "todo"
    CATEGORY = "FM_nodes/StabStitch"

    def todo(
        self,
        video_1: torch.Tensor,
        crop_side: str,
        target_h: int,
        target_w: int,
    ):
        resized_video = crop_or_resize(video_1, target_h, target_w, crop_side)

        return (resized_video,)
