import torch
import torch.nn.functional as F


def ensure_size(tensor: torch.Tensor, size: tuple[int, int] = (128, 128)):
    if tensor.shape[-2:] == size:
        return tensor
    else:
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


def img_to_mask(tensor: torch.Tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device)
    weights = weights.view(1, 3, 1, 1)
    grayscale = torch.sum(tensor * weights, dim=1, keepdim=True)
    return grayscale

