WFEN_MODEL_PATH = "models/wfen/WFEN.pth"
REALVIFORMER_MODEL_PATH = "models/realviformer/weights.pth"
PROPIH_VGG_MODEL_PATH = "models/propih/vgg_normalised.pth"
PROPIH_G_MODEL_PATH = "models/propih/latest_net_G.pth"
VFI_MAMBA_MODELS = ["VFIMamba", "VFIMamba_S"]
VFI_MAMBA_PATHS = {
    "VFIMamba": "models/vfimamba/VFIMamba.pkl",
    "VFIMamba_S": "models/vfimamba/VFIMamba_S.pkl",
}
VFI_MAMBA_DEFAULT = "VFIMamba"

STAB_TEMPORAL_PATH = "models/stabstitch/temporal_warp.pth"
STAB_SPATIAL_PATH = "models/stabstitch/spatial_warp.pth"
STAB_SMOOTH_PATH = "models/stabstitch/smooth_warp.pth"
STAB_CONSTRUCT_MODES = ["FAST", "NORMAL"]
STAB_CONSTRUCT_DEFAULT = "FAST"