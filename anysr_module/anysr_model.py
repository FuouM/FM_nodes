import torch
from .edsr_anysr import make_edsr
from .rdn_anysr import make_rdn


def run_anysr(
    image: torch.Tensor, model_arch: str, model_path: str, scale: float, entire_net=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_max = 4

    model_load = torch.load(model_path)['model']
    model_args = model_load['args']['encoder_spec']['args']
    pretrain_dict = model_load['sd']
    
    print(f"{model_args=}")
    
    image_tensor = image.permute(0, 3, 1, 2).squeeze(dim=0).to(device)
    print(f"{image_tensor.shape=}")
    if model_arch == "edsr":
        model = make_edsr(scale=scale, **model_args)
    elif model_arch == "rdn":
        model = make_rdn(scale=scale, **model_args)
    
    model_dict = model.state_dict()
    model_dict = replace(pretrain_dict, model_dict)
    model.load_state_dict(model_dict, strict=False)
       
    
    # model.load_state_dict(torch.load(model_path)['model']['sd'])

    model = model.to(device)

    h = int(image_tensor.shape[-2] * int(scale))
    w = int(image_tensor.shape[-1] * int(scale))
    scale = h / image_tensor.shape[-2]

    # coord = make_coord((h, w), flatten=False).to(device)
    # cell = torch.ones(1, 2).to(device)
    # cell[:, 0] *= 2 / h
    # cell[:, 1] *= 2 / w

    # cell_factor = max(scale / scale_max, 1)

    # model.encoder.scale = scale
    # model.encoder.scale2 = scale

    pred = model(
        image_tensor.cuda().unsqueeze(0),
    ).squeeze(0)
    print(f"{pred.shape=}")
    pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(3, h, w).cpu()
    print(f"{pred.shape=}")
    pred = pred.permute(1, 2, 0).unsqueeze(dim=0)

    return pred

def replace(pretrain_dict, model_dict):
    new_dict = {}
    for k,v in pretrain_dict.items():
        if 'body.0.' in k[15:]:
            newk = k[:15] + k[15:].replace('body.0','conv1')
            new_dict[newk] = v
        elif 'body.2' in k[15:]:
            newk = k[:15] + k[15:].replace('body.2','conv2')
            new_dict[newk] = v
    pretrain_dict.update(new_dict)
    common_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
    model_dict.update(common_dict)
    return model_dict

def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
