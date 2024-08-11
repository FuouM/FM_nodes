import torch
from torch import nn

from .networks import VGG19HRNet, vgg


class VGG19HRNetModel:
    def __init__(self, vgg_path: str, g_path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_names = ["G"]
        self.netvgg = vgg
        self.netvgg.load_state_dict(torch.load(vgg_path))
        self.netG = VGG19HRNet(self.netvgg)
        self.netG.to(self.device)
        self.netG = nn.DataParallel(self.netG)
        state_dict = torch.load(g_path, map_location=str(self.device))
        self.netG.module.load_state_dict(state_dict, strict=True)
        
    def forward(self, comp, style, mask):
        """Employ generator to generate the output, and calculate the losses for generator"""

        mask_inp = mask / 2 + 0.5

        (
            final_output_1,
            final_output_2,
            final_output_3,
            final_output_4,
            coarse_output_1,
            coarse_output_2,
            coarse_output_3,
            coarse_output_4,
            blend_mask1,
            blend_mask2,
            blend_mask3,
            blend_mask4,
            loss_c,
            loss_s,
        ) = self.netG.forward(comp, style, mask_inp)

        return (
            final_output_1,
            final_output_2,
            final_output_3,
            final_output_4,
        )
