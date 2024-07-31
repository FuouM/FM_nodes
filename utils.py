import torch
import torch.nn.functional as F


def ensure_size(tensor: torch.Tensor, size: tuple[int, int] = (128, 128)):
    if tensor.shape[-2:] == size:
        return tensor
    else:
        return F.interpolate(
            tensor, size=size, mode="bilinear", align_corners=False
        )
