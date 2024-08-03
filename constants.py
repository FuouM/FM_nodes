WFEN_MODEL_PATH = "models/wfen/WFEN.pth"
REALVIFORMER_MODEL_PATH = "models/realviformer/weights.pth"
PROPIH_VGG_MODEL_PATH = "models/propih/vgg_normalised.pth"
PROPIH_G_MODEL_PATH = "models/propih/latest_net_G.pth"
PCSR_MODELS = ["carn", "fsrcnn", "srresnet"]
PCSR_MODEL_PATHS = {
    "carn": "models/pcsr/carn_pcsr/iter_last.pth",
    "fsrcnn": "models/pcsr/fsrcnn_pcsr/iter_last.pth",
    "srresnet": "models/pcsr/srresnet_pcsr/iter_last.pth",
}
PCSR_DEFAULT = "carn"
