import torch
import torch.nn.utils as tutils
from torch import nn

from .models import WFEN


def apply_norm(net: nn.Module, weight_norm_type: str):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if weight_norm_type.lower() == "spectral_norm":
                tutils.spectral_norm(m)
            elif weight_norm_type.lower() == "weight_norm":
                tutils.weight_norm(m)
            else:
                pass


class WFENModel:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.netG = WFEN()
        self.netG.to(self.device)
        self.netG = nn.DataParallel(self.netG)

    def for_load_pretrain_model(self, path):
        print("Loading pretrained model", path)
        weight = torch.load(path)
        self.netG.module.load_state_dict(weight)

    def forward(self, img_LR):
        return self.netG(img_LR)
