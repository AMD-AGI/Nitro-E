# AMD Nitro-E

<div>
  <a href="https://advanced-micro-devices-rocm-blogs--1559.com.readthedocs.build/projects/internal/en/1559/artificial-intelligence/nitro-e/README.html"><img src="https://img.shields.io/badge/ROCm%20Developer%20Blog-red"></a> &ensp;
  <a href="https://huggingface.co/amd/Nitro-E"><img src="https://img.shields.io/badge/Nitro%20E-HuggingFace-yellow"></a> &ensp;
</div>

---
![combined_image_3x3](assets/teaser.png)

## 🔆 Introduction
Nitro-E is a family of text-to-image diffusion models focused on highly efficient training. With just 304M parameters, Nitro-E is designed to be resource-friendly for both training and inference. For training, it only takes 1.5 days on a single node with 8 AMD Instinct&trade; MI300X GPUs. On the inference side, Nitro-E delivers a throughput of **18.8** samples per second (batch size 32, 512px images) a single AMD Instinct MI300X GPU. The distilled version can further increase the throughput to **39.3** samples per second. On a consumer iGPU device Strix Halo, our model can generate a 512px image using only **0.16** second.

This repository provides training and data preparation scripts to reproduce our results. We hope this codebase for efficient diffusion model training enables researchers to iterate faster on ideas and lowers the barrier for independent developers to build custom models.

## 📝 Change Log
- __[2025.10.24]__: 🔥Release [Nitro-E](https://huggingface.co/amd/Nitro-E) Release Nitro-E-512px model, Nitro-E-512px-GRPO post-training GRPO model, Nitro-E-512px-dist distilled model, training and inference code!

## Instruction
### Environment

#### Docker Image

When running on AMD Instinct<sup>TM</sup> GPUs, it is recommended to use the [public PyTorch ROCm images](https://hub.docker.com/r/rocm/pytorch-training/) to get optimized performance out-of-the-box.

```bash
docker pull rocm/pytorch:rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
```

#### Dependencies
```bash
pip install diffusers==0.32.2 transformers==4.49.0 accelerate==1.7.0 wandb torchmetrics pycocotools torchmetrics[image] mosaicml-streaming==0.11.0 beautifulsoup4 tabulate timm==0.9.1 pyarrow einops omegaconf sentencepiece==0.2.0 pandas==2.2.3 alive-progress
```

#### Install Flash attention
```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
MAX_JOBS=`nproc` python setup.py install
```

### Data Processing

The E-MMDiT models were trained on a dataset of ~25M images consisting of both real and synthetic data that are openly available on the internet, including [Segment-Anything-1B](https://ai.meta.com/datasets/segment-anything/), [JourneyDB](https://journeydb.github.io/), and FLUX-generated images using prompts from 
[DiffusionDB](https://github.com/poloclub/diffusiondb) and [DataComp](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B). 

We provide a full pipeline to create the data. 
Please go to `datasets` folder and run:

```bash
cd datasets
bash scripts/get_data_all.sh
``` 

to create the complete version of the dataset.

Or run:

```bash
cd datasets
bash scripts/get_data_partial.sh
``` 
to create a tiny version for testing purposes.


### Model Training

Launch a training session using this script:

```bash
bash scripts/train_512.sh
```

Please modify `configs/accelerate.yaml` accordingly for multi-GPU / multi-node distributed training setup, torch compile, etc., and specific yaml files in `configs` for experimental settings.


### Model Inference

#### Full-step Model

```python
import torch
from core.tools.inference_pipe import init_pipe

device = torch.device('cuda:0')
dtype = torch.bfloat16
resolution = 512
repo_name = "amd/Nitro-E"
ckpt_name = 'Nitro-E-512px.safetensors'

pipe = init_pipe(device, dtype, resolution, repo_name=repo_name, ckpt_name=ckpt_name)
prompt = 'A hot air balloon in the shape of a heart grand canyon'
images = pipe(prompt=prompt, width=resolution, height=resolution, num_inference_steps=20, guidance_scale=4.5).images
```

#### Distilled Model

```python
import torch
from core.tools.inference_pipe import init_pipe

device = torch.device('cuda:0')
dtype = torch.bfloat16
resolution = 512
repo_name = "amd/Nitro-E"
ckpt_name = 'Nitro-E-512px-dist.safetensors'

pipe = init_pipe(device, dtype, resolution, repo_name=repo_name, ckpt_name=ckpt_name)
prompt = 'A hot air balloon in the shape of a heart grand canyon'

images = pipe(prompt=prompt, width=resolution, height=resolution, num_inference_steps=4, guidance_scale=0).images
```


## 🔗 Related Projects
- [Nitro-T](https://github.com/AMD-AGI/Nitro-T): Efficient Training of diffusion models.
- [Nitor-1](https://github.com/AMD-AGI/Nitro-1): One-step distillation of diffusion models.


## License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

This project is licensed under the [MIT License](https://mit-license.org/).
