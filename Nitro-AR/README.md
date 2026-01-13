# AMD Nitro-AR

## 🔆 Introduction

Nitro-AR adapts our previously released Nitro-E diffusion model to a masked autoregressive framework. With optimized sampling strategies such as joint sampling and adversarial fine-tuning, Nitro-AR improves both speed and few-step generation quality, demonstrating the robustness of our architecture across different paradigms.

This repository provides the inference code and model definitions for Nitro-AR.

## 📝 Change Log

- __[2026.1.12]__: 🔥 Initial Release of Nitro-AR inference code and model checkpoints.

## Instruction

### Environment

#### Docker Image

When running on AMD Instinct<sup>TM</sup> GPUs, it is recommended to use the [public PyTorch ROCm images](https://hub.docker.com/r/rocm/pytorch-training/) to get optimized performance out-of-the-box.

```bash
docker pull rocm/pytorch:rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
```

#### Dependencies

Install the required Python packages:

```bash
pip install diffusers==0.32.2 transformers==4.49.0 accelerate==1.7.0 wandb torchmetrics pycocotools torchmetrics[image] mosaicml-streaming==0.11.0 beautifulsoup4 tabulate timm==0.9.1 pyarrow einops omegaconf sentencepiece==0.2.0 pandas==2.2.3 alive-progress ftfy peft safetensors
```

### Model Inference

To run inference, we provide a unified pipeline in `core/tools/inference_pipe.py` and a sample script `test_pipe.py`.

#### Prerequisites

1.  **Checkpoints**: Download the models from [Hugging Face](https://huggingface.co/amd/Nitro-AR).
    - `Nitro-AR-512px-GAN.safetensors` (Adversarial Fine-tuned, 3 steps)
    - `Nitro-AR-512px-Joint-GAN.safetensors` (Joint Sampling + Adversarial Fine-tuned, 6 steps)
    Place them in the `ckpts/` directory.
2.  **Llama-1B**: Ensure the Llama-1B model is available.
3.  **Config**: Verify `configs/config.yaml` and `configs/config_joint.yaml` are correctly set up.

#### Running Inference

We support two optimized model variants: `gan` and `joint_gan`.

You can use the `init_pipe` function to easily load and run these models.

```python
from core.tools.inference_pipe import init_pipe
import torch

device = torch.device('cuda')
dtype = torch.bfloat16

# Initialize pipeline (defaults to 'gan' model, 3 steps)
pipe = init_pipe(device, dtype, model_type='gan')

# Run inference
prompt = "a photo of a dog, with a white background"
output = pipe(prompt=prompt)
output.images[0].save("output_gan.png")

# For Joint-GAN model (6 steps)
pipe_joint = init_pipe(device, dtype, model_type='joint_gan')
output_joint = pipe_joint(prompt=prompt)
output_joint.images[0].save("output_joint_gan.png")
```

See `test_pipe.py` for a complete example.



## 🔗 Related Projects

- [Nitro-E](https://huggingface.co/amd/Nitro-E): Efficient text-to-image diffusion models.
- [Nitro-T](https://github.com/AMD-AGI/Nitro-T): Efficient Training of diffusion models.
- [Nitro-1](https://github.com/AMD-AGI/Nitro-1): One-step distillation of diffusion models.

## License

Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

This project is licensed under the [MIT License](https://mit-license.org/).

