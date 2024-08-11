MODEL_DIR = "models/convir"
SIZE_BASE = "base"
SIZE_SMALL = "small"
SIZE_LARGE = "large"

########### Image Dehazing ###########
DEHAZE_FULL = "haze4k"
DEHAZE_MODELS = [
    "its",
    "ots",
    "densehaze",
    "haze4k",
    "ihaze",
    "nhhaze",
    "ohaze",
    "gta5",
    "nhr",
]

DEHAZE_DEFAULT = "haze4k"

DEHAZE_SIZES = ["small", "base", "large"]


def get_dehaze_model(model_name: str, model_size: str):
    if model_name != DEHAZE_FULL and model_size == "large":
        print(f"Invalid size for model {model_name}. Revert to {SIZE_BASE}")
        model_size = SIZE_BASE
    return f"dehaze/{model_name}-{model_size}.pkl", model_size


########### Image Deraining ###########
DERAIN_MODEL = "deraining"
DERAIN_SIZE = "large"


def get_derain_model():
    return f"{DERAIN_MODEL}.pkl"


########### Image Desnowing ###########
DESNOW_MODELS = ["csd", "snow100k", "srrs"]
DESNOW_DEFAULT = "snow100k"
DESNOW_SIZES = ["small", "base"]


def get_desnow_model(model_name: str, model_size: str):
    return f"desnow/{model_name}-{model_size}.pkl"


########### Defocus Deblurring ###########
DEFOCUS_MODEL = "dpdd"
DEFOCUS_SIZES = ["small", "base", "large"]


def get_defocus_model(model_size: str):
    return f"defocus/{DEFOCUS_MODEL}-{model_size}.pkl"


########### Motion Deblurring ###########
DEBLUR_MODELS = ["gopro", "rsblur"]
DEBLUR_DEFAULT = "gopro"


def get_deblur_model(model_name: str):
    return f"modeblur/convir_{model_name}.pkl"
