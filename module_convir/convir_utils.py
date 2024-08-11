import torch
import torch.nn.functional as F


def preprocess_image(image: torch.Tensor, device, factor=32, min_size=225):
    # [B, H, W, C]
    input_img = image.permute(0, 3, 1, 2).to(device)  # [B, C, H, W]
    _, _, h, w = input_img.shape

    # Resize if either dimension is smaller than min_size
    if h < min_size or w < min_size:
        scale = min_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        input_img = F.interpolate(
            input_img, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        h, w = new_h, new_w

    # Calculate padding
    H = ((h + factor - 1) // factor) * factor
    W = ((w + factor - 1) // factor) * factor

    padh = H - h
    padw = W - w

    # Pad the image
    if padh > 0 or padw > 0:
        input_img = F.pad(input_img, (0, padw, 0, padh), mode="reflect")

    return input_img, h, w
