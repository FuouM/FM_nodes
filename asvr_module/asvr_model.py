import torch
# import tqdm
import torch.nn.functional as F

from .refsrrnn_adists_fgda_only_future import RefsrRNN


class ASVR_Config:
    def __init__(
        self, model_path: str, space_scale=(8, 8), border: bool = False
    ) -> None:
        self.space_scale = space_scale
        self.model_path = model_path
        self.border = border
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ASVR:
    def __init__(self, cfg: ASVR_Config) -> None:
        self.net = RefsrRNN()
        self.net.load_state_dict(torch.load(cfg.model_path))
        self.net = self.net.to(cfg.device)
        self.net.eval()
        self.scale = cfg.space_scale
        self.device = cfg.device

    def inference(self, frames: torch.Tensor):
        # frames: [B, H, W, C]
        print(f"{self.scale=}")
        print(f"{frames.shape=}")
        frames = frames.to(self.device)
        with torch.no_grad():
            scale_h = torch.ones(1).to(self.device) * (1 / float(self.scale[0]))
            scale_w = torch.ones(1).to(self.device) * (1 / float(self.scale[1]))

            hs, hw = 1.0 / scale_h, 1.0 / scale_w
            hs, hw = hs.unsqueeze(-1), hw.unsqueeze(-1)
            kernel = None

            b, h, w, c = frames.shape

            hr_coord = make_coord((h, w)).unsqueeze(0).to(self.device)

            cell = torch.ones(2).unsqueeze(0).to(self.device)
            cell[:, 0] *= 2.0 / h
            cell[:, 1] *= 2.0 / w
            # [B, C, H, W]
            frames = frames.permute(0, 3, 1, 2)

            frames_scaled: list[torch.Tensor] = []
            for frame in frames:
                frames_scaled.append(
                    F.interpolate(
                        frame.unsqueeze(0),
                        (int(scale_h[0].item() * h), int(scale_w[0].item() * w)),
                        mode="bicubic",
                    )
                )

            if kernel is None:
                res = torch.zeros(
                    (
                        1,
                        self.net.num_channels,
                        frames_scaled[0].shape[-2],
                        frames_scaled[0].shape[-1],
                    ),
                    device=frames_scaled[0].device,
                )
                kernel = self.net.kernel_predict(res, hr_coord, cell)

            output = self.net.test_forward(frames_scaled, kernel, hr_coord).squeeze(0) #T,C,H,W
            output = output.permute(0, 2, 3, 1)
        return output


def make_coord(shape):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        # v0, v1 = -1, 1

        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
    ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
    return ret
