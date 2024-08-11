import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .filter import FastGuidedFilter
from .loss import L_TV, L_exp
from .siren import INF


class CoLIE_Config:
    def __init__(
        self,
        down_res: int = 256,
        epochs: int = 100,
        cxt_window: int = 1,
        loss_mean: float = 0.5,
        alpha: float = 1.0,
        beta: float = 20.0,
        gamma: float = 8.0,
        delta: float = 5.0,
    ) -> None:
        self.down_res = down_res
        self.epochs = epochs
        self.cxt_window = cxt_window
        self.loss_mean = loss_mean
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def get_args(self):
        return {
            "down_res": self.down_res,
            "epochs": self.epochs,
            "cxt_window": self.cxt_window,
            "loss_mean": self.loss_mean,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
        }


def run_colie(image: torch.Tensor, cfg: CoLIE_Config):
    # https://github.com/comfyanonymous/ComfyUI/issues/2946#issuecomment-1974060331
    with torch.inference_mode(mode=False):
        # Comfy is [1, H, W, 3]
        # Expect [1, 3, H, W]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)

        tensor_rgb = image.permute(0, 3, 1, 2)

        tensor_hsv = rgb2hsv_torch(tensor_rgb)
        img_v = get_v_component(tensor_hsv)
        img_v_lr = interpolate_image(img_v, cfg.down_res, cfg.down_res)
        coords = get_coords(cfg.down_res, cfg.down_res)
        patches = get_patches(img_v_lr, cfg.cxt_window)

        img_siren = INF(
            patch_dim=cfg.cxt_window**2, num_layers=4, hidden_dim=256, add_layer=2
        )

        img_siren.to(device)

        optimizer = torch.optim.Adam(
            img_siren.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4
        )

        Loss_exp = L_exp(16, cfg.loss_mean)
        Loss_TV = L_TV()

        for epoch in tqdm.tqdm(range(cfg.epochs), desc="Training"):
            img_siren.train()
            optimizer.zero_grad()
            illu_res_lr = img_siren(patches, coords)
            illu_res_lr = illu_res_lr.view(1, 1, cfg.down_res, cfg.down_res)
            illu_lr = illu_res_lr + img_v_lr

            img_v_fixed_lr = (img_v_lr) / (illu_lr + 1e-4)

            loss_spa = (
                torch.mean(torch.abs(torch.pow(illu_lr - img_v_lr, 2))) * cfg.alpha
            )
            loss_tv = Loss_TV(illu_lr) * cfg.beta
            loss_exp = torch.mean(Loss_exp(illu_lr)) * cfg.gamma
            loss_sparsity = torch.mean(img_v_fixed_lr) * cfg.delta

            loss = (
                loss_spa * cfg.alpha
                + loss_tv * cfg.beta
                + loss_exp * cfg.gamma
                + loss_sparsity * cfg.delta
            )

            loss.backward()
            optimizer.step()

    img_v_fixed = filter_up(img_v_lr, img_v_fixed_lr, img_v)
    img_hsv_fixed = replace_v_component(tensor_hsv, img_v_fixed)
    img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)

    return img_rgb_fixed.permute(0, 2, 3, 1)


def get_v_component(img_hsv: torch.Tensor):
    """
    Assumes (1,3,H,W) HSV image.
    """
    return img_hsv[:, -1].unsqueeze(0)


def replace_v_component(img_hsv, v_new):
    """
    Replaces the V component of a HSV image (1,3,H,W).
    """
    img_hsv[:, -1] = v_new
    return img_hsv


def interpolate_image(img, H, W):
    """
    Reshapes the image based on new resolution.
    """
    return F.interpolate(img, size=(H, W))


def get_coords(H, W):
    """
    Creates a coordinates grid for INF.
    """
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W)))
    coords = torch.from_numpy(coords).float().cuda()
    return coords


def get_patches(img, KERNEL_SIZE):
    """
    Creates a tensor where the channel contains patch information.
    """
    kernel = torch.zeros((KERNEL_SIZE**2, 1, KERNEL_SIZE, KERNEL_SIZE)).cuda()

    for i in range(KERNEL_SIZE):
        for j in range(KERNEL_SIZE):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1

    pad = nn.ReflectionPad2d(KERNEL_SIZE // 2)
    im_padded = pad(img)

    extracted = torch.nn.functional.conv2d(im_padded, kernel, padding=0).squeeze(0)

    return torch.movedim(extracted, 0, -1)


def filter_up(x_lr, y_lr, x_hr, r=1):
    """
    Applies the guided filter to upscale the predicted image.
    """
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (-torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb
