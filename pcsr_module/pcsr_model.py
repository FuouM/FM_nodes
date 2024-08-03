import torch

from .other_utils import make_coord
from .models import make
from .core import imresize


def run_pcsr(
    image: torch.Tensor,
    model_path: str,
    scale=4,
    k=0.0,
    pixel_batch_size=300000,
    adaptive=False,
    no_refinement=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sv_file = torch.load(model_path)
    model = make(sv_file["model"], load_sd=True).to(device)
    model.eval()

    rgb_mean = torch.tensor([0.4488, 0.4371, 0.4040], device=device).view(1, 3, 1, 1)
    rgb_std = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1)

    image_tensor = image.permute(0, 3, 1, 2)

    # center-crop hr to scale-divisible size
    H, W = image_tensor.shape[-2:]
    H, W = H // scale * scale, W // scale * scale
    h, w = H // scale, W // scale
    hr = center_crop(image_tensor, (H, W)).to(device)
    lr = imresize(hr, sizes=(h, w))  # [0,1]
    inp_lr = (lr - rgb_mean) / rgb_std

    # model prediction
    coord = make_coord((H, W), flatten=True, device="cuda").unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / H
    cell[:, :, 1] *= 2 / W
    
    pred, flag = model(
        inp_lr,
        coord=coord,
        cell=cell,
        scale=scale,
        k=k,
        pixel_batch_size=pixel_batch_size,
        adaptive_cluster=adaptive,
        refinement=not no_refinement,
    )
    pred = pred.transpose(1, 2).view(-1, 3, H, W)
    pred = pred * rgb_std + rgb_mean
    flag = flag.view(-1, 1, H, W).repeat(1, 3, 1, 1)

    print(f"{pred.shape=}")
    print(f"{flag.shape=}")

    return pred.permute(0, 2, 3, 1), flag.permute(0, 2, 3, 1)


def center_crop(img, size):
    h, w = img.shape[-2:]
    cut_h, cut_w = h - size[0], w - size[1]

    lh = cut_h // 2
    rh = h - (cut_h - lh)
    lw = cut_w // 2
    rw = w - (cut_w - lw)

    img = img[:, :, lh:rh, lw:rw]
    return img

