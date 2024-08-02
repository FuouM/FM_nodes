# FM_nodes

A collection of ComfyUI nodes.

Click name to jump to workflow
1. [**WFEN Face Restore**](#wfen). Paper: [Efficient Face Super-Resolution via Wavelet-based Feature Enhancement Network](https://github.com/PRIS-CV/WFEN)
2. [**RealViformer**](#realviformer) - Paper: [Investigating Attention for Real-World Video Super-Resolution](https://github.com/Yuehan717/RealViformer)
3. [**ProPIH**](#propih). Paper: [Progressive Painterly Image Harmonization from Low-level Styles to High-level Styles](https://github.com/bcmi/ProPIH-Painterly-Image-Harmonization)
4. [**CoLIE**](#colie). Paper: [Fast Context-Based Low-Light Image Enhancement via Neural Implicit Representations](https://github.com/ctom2/colie)
   
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
