# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

import torch
from core.models.emmdit_pipeline import EMMDiTPipeline
from transformers import AutoModelForCausalLM,AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from diffusers import AutoencoderDC
from core.models.transformer_emmdit import EMMDiTTransformer
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import os

def init_pipe(device, dtype, resolution, repo_name, ckpt_name, ckpt_path_grpo=None):
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    text_encoder = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', torch_dtype=dtype)
    text_encoder.to(device)

    scheduler = FlowMatchEulerDiscreteScheduler(1000)

    vae = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=dtype).to(device).eval()
    if resolution == 1024:
        transformer = EMMDiTTransformer(sample_size=resolution//32)
    else:
    	transformer = EMMDiTTransformer(sample_size=resolution//32, use_sub_attn=False)
    transformer.eval() 
    state_dict = load_file(hf_hub_download(repo_name, ckpt_name)) 
    transformer.load_state_dict(state_dict)
    if ckpt_path_grpo is not None:
        from peft import PeftModel
        transformer = PeftModel.from_pretrained(transformer, os.path.join(snapshot_download(repo_id=repo_name),ckpt_path_grpo), is_trainable=False)
        transformer = transformer.merge_and_unload()
        
    transformer = transformer.to(device).to(dtype)    
    pipe = EMMDiTPipeline(tokenizer, text_encoder, vae, transformer, scheduler)
    
    return pipe
    
