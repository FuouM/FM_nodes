import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import tqdm
from . import torch_tps_transform
from .smooth_network import SmoothNet, build_SmoothNet
from .spatial_network import SpatialNet, build_SpatialNet
from .temporal_network import TemporalNet, build_TemporalNet
from . import torch_tps_transform_point

grid_h = 6
grid_w = 8

img_h = 360
img_w = 480


def video_preprocess(src_video: torch.Tensor, h: int, w: int):
    # [B, H, W, C] [0, 1]
    tensor = src_video.permute(0, 3, 1, 2)  # [B, C, H, W]
    tensor = tensor[:, [2, 1, 0], :, :]  # RGB -> BGR
    tensor = tensor * 2 - 1  # [-1, 1]
    tensor = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
    return tensor


def video_to_list_ts(video: torch.Tensor):
    video_list = [tensor.unsqueeze(0) for tensor in video]
    return video_list


def stabstitch_inference(
    src_video_1: torch.Tensor,
    src_video_2: torch.Tensor,
    spatial_path: str,
    temporal_path: str,
    smooth_path: str,
    device,
):
    spatial_net = SpatialNet().to(device)
    temporal_net = TemporalNet().to(device)
    smooth_net = SmoothNet().to(device)

    spatial_net.load_state_dict(torch.load(spatial_path)["model"])
    temporal_net.load_state_dict(torch.load(temporal_path)["model"])
    smooth_net.load_state_dict(torch.load(smooth_path)["model"])

    spatial_net.eval()
    temporal_net.eval()
    smooth_net.eval()

    video_1 = video_preprocess(src_video_1, img_h, img_w).to(device)
    video_2 = video_preprocess(src_video_2, img_h, img_w).to(device)
    video_list_1 = video_to_list_ts(video_1)
    video_list_2 = video_to_list_ts(video_2)

    num_frames = len(video_list_1)
    smotion_tensor_list = []
    omask_tensor_list = []

    # motion estimation
    for k in tqdm.tqdm(range(num_frames), desc="Motion estimating"):
        # step 1: spatial warp
        with torch.no_grad():
            spatial_batch_out = build_SpatialNet(
                spatial_net, video_list_1[k], video_list_2[k]
            )
        smotion = spatial_batch_out["motion"]
        omask = spatial_batch_out["overlap_mesh"]
        smotion_tensor_list.append(smotion)
        omask_tensor_list.append(omask)

    # step 2: temporal warp
    with torch.no_grad():
        temporal_batch_out = build_TemporalNet(temporal_net, video_list_2)
    tmotion_tensor_list = temporal_batch_out["motion_list"]

    ##############################################
    #############   data preparation  ############
    # converting tmotion (t-th frame) into tsmotion ( (t-1)-th frame )
    rigid_mesh = get_rigid_mesh(1, img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

    rigid_mesh = get_rigid_mesh(1, img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

    smesh_list = []
    tsmotion_list = []

    for k in tqdm.tqdm(range(len(tmotion_tensor_list)), desc="Temporal calculating"):
        smotion = smotion_tensor_list[k]
        smesh = rigid_mesh + smotion
        if k == 0:
            tsmotion = smotion.clone() * 0

        else:
            smotion_1 = smotion_tensor_list[k - 1]
            smesh_1 = rigid_mesh + smotion_1
            tmotion = tmotion_tensor_list[k]
            tmesh = rigid_mesh + tmotion
            norm_smesh_1 = get_norm_mesh(smesh_1, img_h, img_w)
            norm_tmesh = get_norm_mesh(tmesh, img_h, img_w)
            tsmesh = torch_tps_transform_point.transformer(
                norm_tmesh, norm_rigid_mesh, norm_smesh_1
            )
            tsmotion = recover_mesh(tsmesh, img_h, img_w) - smesh
        # append
        smesh_list.append(smesh)
        tsmotion_list.append(tsmotion)

    # step 3: smooth warp
    ori_mesh = 0
    target_mesh = 0

    for k in tqdm.tqdm(range(len(tmotion_tensor_list) - 6), "Smooth meshing"):
        # get sublist and set the first element to 0
        tsmotion_sublist = tsmotion_list[k : k + 7]
        tsmotion_sublist[0] = smotion_tensor_list[k] * 0

        with torch.no_grad():
            smooth_batch_out = build_SmoothNet(
                smooth_net,
                tsmotion_sublist,
                smesh_list[k : k + 7],
                omask_tensor_list[k : k + 7],
            )

        _ori_mesh = smooth_batch_out["ori_mesh"]
        _target_mesh = smooth_batch_out["target_mesh"]

        if k == 0:
            ori_mesh = _ori_mesh
            target_mesh = _target_mesh

        else:
            ori_mesh = torch.cat((ori_mesh, _ori_mesh[:, -1, ...].unsqueeze(1)), 1)
            target_mesh = torch.cat(
                (target_mesh, _target_mesh[:, -1, ...].unsqueeze(1)), 1
            )

    stable_list, mesh_list_2, img_2_list, out_width, out_height = get_stable_sqe(
        video_list_1, video_list_2, target_mesh
    )

    stable_list = [stable.unsqueeze(0)[:, [2, 1, 0], :, :] for stable in stable_list]

    print(f"{stable_list[0].shape=}")  # ([3, 380, 757])
    print(f"{img_2_list[0].shape=}")  # ([3, 380, 757])
    if mesh_list_2:
        print(f"{mesh_list_2[0].shape=}")

    return stable_list, img_2_list, mesh_list_2


def linear_blender(ref, tgt, ref_m, tgt_m, mask=False):
    blur = GaussianBlur(kernel_size=(21, 21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())

    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_).cuda()
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (
        proj_val.max() - proj_val.min() + 1e-3
    )

    mask1 = (
        blur(ref_m_ + (1 - ovl_mask) * ref_m[:, 0].unsqueeze(1)) * ref_m + ref_m_
    ).clamp(0, 1)
    if mask:
        return mask1

    mask2 = (1 - mask1) * tgt_m
    stit = ref * mask1 + tgt * mask2

    return stit


def recover_mesh(norm_mesh, height, width):
    # from [bs, pn, 2] to [bs, grid_h+1, grid_w+1, 2]

    batch_size = norm_mesh.size()[0]
    mesh_w = (norm_mesh[..., 0] + 1) * float(width) / 2.0
    mesh_h = (norm_mesh[..., 1] + 1) * float(height) / 2.0
    mesh = torch.stack([mesh_w, mesh_h], 2)  # [bs,(grid_h+1)*(grid_w+1),2]

    return mesh.reshape([batch_size, grid_h + 1, grid_w + 1, 2])


def get_rigid_mesh(batch_size, height, width):
    ww = torch.matmul(
        torch.ones([grid_h + 1, 1]),
        torch.unsqueeze(torch.linspace(0.0, float(width), grid_w + 1), 0),
    )
    hh = torch.matmul(
        torch.unsqueeze(torch.linspace(0.0, float(height), grid_h + 1), 1),
        torch.ones([1, grid_w + 1]),
    )
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)), 2)  # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt


def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[..., 0] * 2.0 / float(width) - 1.0
    mesh_h = mesh[..., 1] * 2.0 / float(height) - 1.0
    norm_mesh = torch.stack([mesh_w, mesh_h], 3)  # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2])  # bs*-1*2


# bs, T, h, w, 2  smooth_path
# def get_stable_sqe(img2_list, ori_mesh):
#     batch_size, _, img_h, img_w = img2_list[0].shape

#     rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
#     norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

#     stable_list = []
#     for i in range(len(img2_list)):
#         mesh = ori_mesh[:, i, :, :, :]
#         norm_mesh = get_norm_mesh(mesh, img_h, img_w)
#         img2 = (img2_list[i].cuda() + 1) / 2
#         print(img2)
#         mask = torch.ones_like(img2).cuda()
#         img2_warp = torch_tps_transform.transformer(
#             torch.cat([img2, mask], 1), norm_mesh, norm_rigid_mesh, (img_h, img_w)
#         )
#         stable_list.append(img2_warp)

#     return stable_list


# FAST
# NORMAL
def get_stable_sqe(img1_list, img2_list, ori_mesh, mode="FAST", do_linear_blend=False):
    batch_size, _, img_h, img_w = img2_list[0].shape

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

    width_max = torch.max(torch.max(ori_mesh[..., 0]))
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(ori_mesh[..., 0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(ori_mesh[..., 1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(ori_mesh[..., 1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    width_min = width_min.int()
    width_max = width_max.int()
    height_min = height_min.int()
    height_max = height_max.int()
    out_width = width_max - width_min + 1
    out_height = height_max - height_min + 1

    print(out_width)
    print(out_height)

    img1_warp = torch.zeros((1, 3, out_height, out_width)).cuda()

    stable_list = []
    mask_list_2 = []
    img_2_list = []
    # mesh_tran_list = []
    for i in range(len(img2_list)):
        mesh = ori_mesh[:, i, :, :, :]
        mesh_trans = torch.stack(
            [mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3
        )
        norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
        img2 = (img2_list[i].cuda() + 1) / 2

        # mode = 'FAST': use F.grid_sample to interpolate. It's fast, but may produce thin black boundary.
        # mode = 'NORMAL': use our implemented interpolation function. It's a bit slower, but avoid the black boundary.
        img2_warp = torch_tps_transform.transformer(
            img2, norm_mesh, norm_rigid_mesh, (out_height, out_width), mode=mode
        )
        img_2_list.append(img2_warp)
        mask2 = torch_tps_transform.transformer(
            torch.ones_like(img2).cuda(),
            norm_mesh,
            norm_rigid_mesh,
            (out_height, out_width),
        )
        mask_list_2.append(mask2)

        if not do_linear_blend:
            # average blending
            img1_warp[
                :,
                :,
                int(0 - height_min) : int(0 - height_min) + 360,
                int(0 - width_min) : int(0 - width_min) + 480,
            ] = (img1_list[i].cuda() + 1) / 2

            comp_1 = img1_warp[0] / (img1_warp[0] + img2_warp[0] + 1e-6)
            comp_2 = img2_warp[0] / (img1_warp[0] + img2_warp[0] + 1e-6)

            ave_fusion = img1_warp[0] * comp_1 + img2_warp[0] * comp_2
        else:
            # linear blending
            img1_warp[
                :,
                int(0 - height_min) : int(0 - height_min) + 360,
                int(0 - width_min) : int(0 - width_min) + 480,
            ] = (img1_list[i][0] + 1) / 2
            mask1 = img1_warp.clone()
            mask1[
                :,
                int(0 - height_min) : int(0 - height_min) + 360,
                int(0 - width_min) : int(0 - width_min) + 480,
            ] = 1

            ave_fusion = linear_blender(
                img1_warp.unsqueeze(0), img2_warp, mask1.unsqueeze(0), mask2
            )
            ave_fusion = ave_fusion[0]

        stable_list.append(ave_fusion)

    return stable_list, mask_list_2, img_2_list, out_width, out_height
