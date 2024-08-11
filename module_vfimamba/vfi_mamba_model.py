import math

import torch
import tqdm
from comfy.utils import ProgressBar

from . import config as cfg
from .padder import InputPadder
from .Trainer_finetune import Model


def run_vfi_mamba(
    src_video: torch.Tensor,
    model_name: str,
    model_path: str,
    vfi_scale=2,
    scale: float = 0.0,
):
    print(f"{model_name=}")
    print(f"{model_path=}")
    TTA = False
    cfg.MODEL_CONFIG["LOGNAME"] = model_name
    if model_name == "VFIMamba":
        TTA = True
        cfg.MODEL_CONFIG["MODEL_ARCH"] = cfg.init_model_config(
            F=32, depth=[2, 2, 2, 3, 3]
        )
    else:
        cfg.MODEL_CONFIG = cfg.MODEL_CONFIG_DEFAULT.copy()
    print()
    model = Model(-1, cfg.MODEL_CONFIG)
    model.load_model(model_path=model_path)
    model.eval()
    model.device()

    src_video = src_video.permute(0, 3, 1, 2).to("cuda")

    out_frames: list[torch.Tensor] = []

    pbar = ProgressBar(src_video.shape[0])
    for i in tqdm.tqdm(range(src_video.shape[0] - 1), "Generating frames"):
        frame_a = src_video[i].unsqueeze(dim=0)
        frame_b = src_video[i + 1].unsqueeze(dim=0)

        padder = InputPadder(frame_a.shape, divisor=32)
        I0_, I2_ = padder.pad(frame_a, frame_b)

        if vfi_scale == 2:
            out = model.inference(I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=scale)
            mid_frame = padder.unpad(out)
            out_frames.append(frame_a)
            out_frames.append(mid_frame)
        else:
            frames = _recursive_generator(
                model=model,
                TTA=TTA,
                scale=scale,
                frame1=I0_,
                frame2=I2_,
                num_recursions=int(math.log2(vfi_scale)),
                index=vfi_scale // 2,
            )
            frames = sorted(frames, key=lambda x: x[1])
            for pred, _ in frames:
                out_frames.append(padder.unpad(pred))
        pbar.update_absolute(i, src_video.shape[0])

    out_frames.append(src_video[-1].unsqueeze(0))

    out_tensor = torch.cat(out_frames, dim=0).permute(0, 2, 3, 1)
    print(f"{out_tensor.shape=}")
    return out_tensor


def _recursive_generator(
    model, TTA, scale, frame1: torch.Tensor, frame2: torch.Tensor, num_recursions, index
):
    if num_recursions == 0:
        return [(frame1, index)]
    else:
        mid_frame = model.inference(
            frame1, frame2, True, TTA=TTA, fast_TTA=TTA, scale=scale
        )
        id = 2 ** (num_recursions - 1)
        return _recursive_generator(
            model, TTA, scale, frame1, mid_frame, num_recursions - 1, index - id
        ) + _recursive_generator(
            model, TTA, scale, mid_frame, frame2, num_recursions - 1, index + id
        )
