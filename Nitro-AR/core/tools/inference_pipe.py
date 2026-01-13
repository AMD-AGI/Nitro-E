# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderDC
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import os

class NitroARPipelineOutput:
    def __init__(self, images):
        self.images = images

class NitroARPipeline:
    def __init__(self, tokenizer, text_encoder, vae, transformer, config, model_type, device, dtype):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.config = config
        self.model_type = model_type
        self.device = device
        self.dtype = dtype
        
        # Set default steps based on model type
        if model_type == 'gan':
            self.num_diff_steps = 3
        elif model_type == 'joint_gan':
            self.num_diff_steps = 6
        else:
             # Fallback if someone manually passes something else, though init_pipe restricts it.
            self.num_diff_steps = 20

    @torch.no_grad()
    def __call__(self, prompt, num_diff_steps=None, cfg=4.0, width=512, height=512):
        # width and height are currently determined by model config (latent_size=16 -> 512px)
        
        if num_diff_steps is None:
            num_diff_steps = self.num_diff_steps
            
        inputs = self.tokenizer([prompt], return_tensors="pt", padding='max_length', max_length=self.config.model.caption_max_seq_length, truncation=True)
        inputs.to(self.device)
        txt_emb = self.text_encoder(**inputs, output_hidden_states=True)['hidden_states'][-1]
        
        # Unconditional embedding
        inputs_uncond = self.tokenizer([''], return_tensors="pt", padding='max_length', max_length=self.config.model.caption_max_seq_length, truncation=True)
        inputs_uncond.to(self.device)
        uncond_txt_emb = self.text_encoder(**inputs_uncond, output_hidden_states=True)['hidden_states'][-1]

        if self.model_type == 'joint_gan':
            # Joint sampling logic
            # Step 1: Generate global token
            gen_global_token, _ = self.transformer.sample_tokens(
                txt_emb, uncond_txt_emb,
                num_iter=1, cfg=cfg, progress=False, pred_global_token=True, 
                diff_steps=num_diff_steps
            )
            
            # Step 2: Generate sample
            gen_sample, _ = self.transformer.sample_tokens(
                txt_emb, uncond_txt_emb,
                num_iter=1, cfg=cfg, progress=True, 
                global_token=gen_global_token,
                global_token_dependency=False,
                cfg_schedule='constant',
                diff_steps=num_diff_steps
            )
        else:
            # Standard sampling logic (for 'gan' and potentially legacy if forced)
            gen_sample = self.transformer.sample_tokens(
                txt_emb, uncond_txt_emb,
                num_iter=20,
                cfg=cfg, progress=True, diff_steps=num_diff_steps
            )

        gen_sample = gen_sample.reshape(-1, 16, 16, 32)
        gen_sample = torch.permute(gen_sample, [0, 3, 1, 2])
        y = self.vae.decode(gen_sample.to(self.device)/self.config.training.scaling_factor).sample
        
        image = y * 0.5 + 0.5
        image = torch.clamp(image, 0, 1)
        image = image.to(torch.float32)
        
        # Convert to PIL images
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        pil_images = [to_pil(img) for img in image]
        
        return NitroARPipelineOutput(images=pil_images)

def init_pipe(device, dtype, model_type='gan', ckpt_path=None):
    """
    Initialize the NitroAR pipeline.
    
    Args:
        device: torch device
        dtype: torch dtype
        model_type: One of ['gan', 'joint_gan']
        ckpt_path: Path to checkpoint. If None, uses default relative paths.
    """
    
    # Defaults
    if ckpt_path is None:
        if model_type == 'gan':
            ckpt_path = 'ckpts/Nitro-AR-512px-GAN.safetensors'
        elif model_type == 'joint_gan':
            ckpt_path = 'ckpts/Nitro-AR-512px-Joint-GAN.safetensors'
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported types: ['gan', 'joint_gan']")

    # Config selection
    if model_type == 'joint_gan':
        config_path = "configs/config_joint.yaml"
    else:
        config_path = "configs/config.yaml"
        
    cfg = OmegaConf.load(config_path)
    
    # Model Initialization
    if model_type == 'joint_gan':
        from core.models.emmdit_joint import MMDiTTransformer2DModel
        # Joint model params
        model = MMDiTTransformer2DModel(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            sample_size=cfg.model.latent_size,
            patch_size=cfg.model.patch_size,
            caption_channels=cfg.model.caption_channels,
            qk_norm='rms_norm',
            repa_depth=cfg.model.repa_depth,
            projector_dim=cfg.model.projector_dim,
            z_dims=cfg.model.z_dims,
            diffloss_d=cfg.model.diffloss_d,
            diffloss_w=cfg.model.diffloss_w,
            num_sampling_steps=cfg.model.num_sampling_steps,
            diffusion_batch_mul=cfg.model.diffusion_batch_mul,
            shift=cfg.training.fm_shift,
            mask_ratio_min=cfg.training.mask_ratio_min,
            global_token_type=cfg.model.global_token_type,
            global_head=cfg.model.global_head
        )
    else:
        from core.models.emmdit import MMDiTTransformer2DModel
        model = MMDiTTransformer2DModel(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            sample_size=cfg.model.latent_size,
            patch_size=cfg.model.patch_size,
            caption_channels=cfg.model.caption_channels,
            qk_norm='rms_norm',
            repa_depth=cfg.model.repa_depth,
            projector_dim=cfg.model.projector_dim,
            z_dims=cfg.model.z_dims,
            diffloss_d=cfg.model.diffloss_d,
            diffloss_w=cfg.model.diffloss_w,
            num_sampling_steps=cfg.model.num_sampling_steps,
            diffusion_batch_mul=cfg.model.diffusion_batch_mul,
            shift=cfg.training.fm_shift,
            mask_ratio_min=cfg.training.mask_ratio_min
        )

    # Load weights
    print(f"Loading checkpoint from {ckpt_path}...")
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)
    
    model = model.to(dtype)
    model.eval()
    model = model.to(device)

    # Tokenizer & Text Encoder
    llama_path = cfg.llama_path
    print(f"Loading Llama from {llama_path}...")
    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    text_encoder = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.to(device)
    
    # VAE
    print("Loading VAE...")
    dc_ae = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=dtype).to(device).eval()

    return NitroARPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=dc_ae,
        transformer=model,
        config=cfg,
        model_type=model_type,
        device=device,
        dtype=dtype
    )
