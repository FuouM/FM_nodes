# FM_nodes

A collection of ComfyUI nodes.

Click name to jump to workflow
1. [**WFEN Face Restore**](#wfen). Paper: [Efficient Face Super-Resolution via Wavelet-based Feature Enhancement Network](https://github.com/PRIS-CV/WFEN)
2. [**RealViformer**](#realviformer) - Paper: [Investigating Attention for Real-World Video Super-Resolution](https://github.com/Yuehan717/RealViformer)
3. [**ProPIH**](#propih). Paper: [Progressive Painterly Image Harmonization from Low-level Styles to High-level Styles](https://github.com/bcmi/ProPIH-Painterly-Image-Harmonization)
4. [**CoLIE**](#colie). Paper: [Fast Context-Based Low-Light Image Enhancement via Neural Implicit Representations](https://github.com/ctom2/colie)
5. [**VFIMamba**](#vfimamba). Paper: [Video Frame Interpolation with State Space Models](https://github.com/MCG-NJU/VFIMamba)
6. [**ConvIR**](#convir) [Revitalizing Convolutional Network for Image Restoration](https://github.com/c-yn/ConvIR)

## Workflows

### WFEN

Download the model [here](https://github.com/PRIS-CV/WFEN?tab=readme-ov-file#getting-started) and place it in `models/wfen/WFEN.pth`.

[workflow_wfen_facecrop.json](workflow/workflow_wfen_facecrop.json)

![wfen_facecrop](workflow/wfen_facecrop.png)

### RealViformer

Download the model [here](https://github.com/Yuehan717/RealViformer?tab=readme-ov-file#usage) and place it in `models/realviformer/weights.pth`.

[workflow_realviformer.json](workflow/workflow_realviformer.json)

![realviformer_example](example_realviformer.png)

(Not a workflow-embedded image)

https://github.com/user-attachments/assets/e89003c0-7be5-4263-b281-fd609807cea1

RealViFormer upscale example

### ProPIH

Download the `vgg_normalised.pth` model in the [Installation section](https://github.com/bcmi/ProPIH-Painterly-Image-Harmonization?tab=readme-ov-file#installation) and `latest_net_G.pth` in the [Train/Test section](https://github.com/bcmi/ProPIH-Painterly-Image-Harmonization?tab=readme-ov-file#propih-traintest)

```
models/propih/vgg_normalised.pth
models/propih/latest_net_G.pth
```

[workflow_propih.json](workflow/workflow_realviformer.json)

![propih](workflow/propih.png)

### CoLIE

No model needed to be downloaded. Lower `loss_mean` seems to result in brighter images. Node works with image and batched/video.

[workflow_colie_lowlight.json](workflow/workflow_colie_lowlight.json)

![colie_lowlight](workflow/colie_lowlight.png)

### VFIMamba

Download the models from the [huggingface page](https://huggingface.co/MCG-NJU/VFIMamba/tree/main/ckpt)

```
models/vfimamba/VFIMamba_S.pkl
models/vfimamba/VFIMamba.pkl
```

You will need to install `mamba-ssm`, which does not have a prebuilt Windows binary. You will need:
1. [triton](https://github.com/triton-lang/triton). Prebuilt for `Python 3.10 and 3.11` can be found here: https://github.com/triton-lang/triton/issues/2881 - https://huggingface.co/madbuda/triton-windows-builds/tree/main
2. [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d). Follow this post: https://github.com/NVlabs/MambaVision/issues/14#issuecomment-2232581078
3. [mamba-ssm](https://github.com/state-spaces/mamba). Follow this tutorial: https://blog.csdn.net/yyywxk/article/details/140420538. Fork that followed all the steps: https://github.com/FuouM/mamba-windows-build

I've built `mamba-ssm` for `Python 3.11, torch 2.3.0+cu121`, which can be obtained here: https://huggingface.co/FuouM/mamba-ssm-windows-builds/tree/main

To install, `pip install [].whl`

[workflow_vfi_mamba.json](workflow/workflow_vfi_mamba.json)

![example_vfi_mamba](example_vfi_mamba.png)

(Not a workflow-embedded image)

https://github.com/user-attachments/assets/be263cc3-a104-4262-899b-242e9802719e

VFIMamba Example (top: Original, bottom: 5X, 20FPS)

### ConvIR

