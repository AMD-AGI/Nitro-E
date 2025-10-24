# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

import os
from argparse import ArgumentParser
import sys
import subprocess
from streaming.base import MDSWriter
import pyarrow.parquet as pq
import torch
from diffusers import FluxPipeline, AutoencoderDC, FlowMatchEulerDiscreteScheduler
from torchvision import transforms
from transformers import AutoModelForCausalLM,AutoTokenizer
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.models.emmdit_pipeline import EMMDiTPipeline

def get_image_transform(resize=1024):
    return transforms.Compose([
        transforms.Resize(
            resize,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args=None) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        required=True,
        help="Directory to store mds shards.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=1,
        help="batch size",
    )     
    parser.add_argument(
        "--imgdim",
        type=int,
        default=512,
        choices=[512,1024],
        help="image px",
    )
    parser.add_argument(
        "--download_mode",
        type=str,
        default='partial',
        choices=['partial','total'],
        help="download datasets range",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=['ucsc','diffusiondb'],
        help="prompt parque file",
    ) 
    parser.add_argument(
        '--text_encoder',
        default=False,
        action='store_true',
        help='If True, write text encode to mds for text encoding.',
    )
    parser.add_argument(
        '--dino_model',
        default=False,
        action='store_true',
        help='If True, write dino feature to mds for dino encoding.',
    )
    
    args = parser.parse_args()
    return args


def encoding_func(pipe, prompt, device):
    prompt = pipe._text_preprocessing(prompt, clean_caption=True)
    text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
    
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.to(device)

    prompt_embeds = pipe.text_encoder(text_input_ids.to(device), 
                                attention_mask=prompt_attention_mask, 
                                output_hidden_states=True)['hidden_states'][-1]    
            
    prompt_embeds = prompt_embeds.squeeze().to(torch.float16).cpu()
    atten_mask = prompt_attention_mask.squeeze().cpu()
    #return {'encoder_hidden_states': prompt_embeds, 
    #        'encoder_attention_mask': atten_mask}
    return prompt_embeds, atten_mask


def write_df(args: ArgumentParser ):
    print(args.imgdim)
    from core.models.transformer_emmdit import EMMDiTTransformer
    if args.text_encoder:
        if args.imgdim==512:
            transformer = EMMDiTTransformer(
                    in_channels=32,
                    out_channels=32,
                    sample_size=16,
                    patch_size=1,
                    caption_channels=2048,
                    qk_norm='rms_norm',
                )
        elif args.imgdim==1024:
            transformer = EMMDiTTransformer(
                    in_channels=32,
                    out_channels=32,
                    sample_size=32,
                    patch_size=1,
                    caption_channels=2048,
                    qk_norm='rms_norm',
                    use_sub_attn = False,
                ) 

        vae = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=torch.bfloat16).to(device).eval()
        vae = torch.compile(vae)

        tokenizer = AutoTokenizer.from_pretrained('/cache/Llama-3.2-1B/')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        text_encoder = AutoModelForCausalLM.from_pretrained('/cache/Llama-3.2-1B/', torch_dtype=torch.bfloat16)
        text_encoder.to(device)
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        from core.models.transformer_emmdit import EMMDiTTransformer
                    
        pipe_emmdit = EMMDiTPipeline(tokenizer, text_encoder, vae, transformer, scheduler)

    if args.dino_model:
        dinoencoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
        del dinoencoder.head
        dinoencoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(dinoencoder.pos_embed.data, [16, 16],)
        dinoencoder.head = torch.nn.Identity()
        dinoencoder = dinoencoder.to(device).eval()
        transforms_dino = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])


    if args.imgdim == 512:  
        columns = {
            "jpg": "jpeg",
            "caption": "str",
            'latents_512': 'bytes',
        }
    elif args.imgdim == 1024:
        columns = {
        'caption': 'str',
        'latents_1024': 'bytes',
        }
    
    if args.dino_model:
        columns['dino_feat'] = 'bytes'
    if args.text_encoder:
        columns['text_feat'] = 'bytes'
        columns['text_mask'] = 'bytes'
        
    mds_folder_path = os.path.join(args.local_mds_dir)

    writer = MDSWriter(
        out=mds_folder_path,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    ) 

    if args.imgdim == 512:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    elif args.imgdim == 1024:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    
    pipe.to(device)
    pipe.transformer = torch.compile(pipe.transformer)

    pipe.set_progress_bar_config(disable=True)
    vae = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=torch.float32).to(device).eval()
    vae = torch.compile(vae) 


    image_transform = get_image_transform(resize=args.imgdim)

    pr_select=[]
    if args.prompt_type == 'ucsc':
        import pandas as pd
        import glob
        dfs = glob.glob('UCSC_VLAA/*.parquet')
        dfs = sorted(dfs)
   
        if args.download_mode == 'partial':
            for idx,fs in enumerate(dfs):
                metadata_df = pd.read_parquet(fs)
                for i in range(256):
                    pr_select.append(metadata_df['re_caption'][i])
        elif args.download_mode == 'total':
            for idx,fs in enumerate(dfs):
                metadata_df = pd.read_parquet(fs)
                for i in range(len(metadata_df)):
                    pr_select.append(metadata_df['re_caption'][i])


    elif args.prompt_type == 'diffusiondb':
        from urllib.request import urlretrieve
        import pandas as pd
        table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata-large.parquet'
        urlretrieve(table_url, 'metadata-large.parquet')
        metadata_df = pd.read_parquet('metadata-large.parquet')
        if args.download_mode == 'partial':
            for i in range(256):
                pr_select.append(metadata_df['prompt'][i])
        elif args.download_mode == 'total':
            for i in range(len(metadata_df)):
                pr_select.append(metadata_df['prompt'][i])

    bs=args.bs
    for i in range(0,len(pr_select)-len(pr_select)%bs, bs):
        pr = pr_select[i:i+bs]
        
        if args.imgdim==512:
            images = pipe(pr, height=512,width=512,guidance_scale=3.5,num_inference_steps=20,max_sequence_length=512).images
            
            image_tensors = []
            for img in images:
                img = img.convert('RGB')
                tensor1 = image_transform(img).to(device)
                image_tensors.append(tensor1)
            image_batch = torch.stack(image_tensors)
            with torch.no_grad():
                latents_512 = vae.encode(image_batch.to(torch.float32)).latent.to(torch.float16)
                
            latents_512 = latents_512.detach().clone().cpu().numpy()
            
            ##extract dino feat
            if args.dino_model:
                image_dino_tensors = []
                for img in images:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    tensor2 = transforms_dino(img).to(device)
                    image_dino_tensors.append(tensor2)
                image_dino_batch = torch.stack(image_dino_tensors)
                
            
                with torch.no_grad():
                    z = dinoencoder.forward_features(image_dino_batch)
                    z = z['x_norm_patchtokens']
                    
                z = z.cpu().half().numpy()                    
            ##extract dino feat
                  
            
            for idx in range(latents_512.shape[0]):
                mds_sample = {
                    "jpg": images[idx],
                    "caption": pr[idx],
                    'latents_512': latents_512[idx].tobytes(),
                    }
                
                if args.dino_model:
                    mds_sample['dino_feat'] = z[idx].tobytes()
                    
                if args.text_encoder:
                    embs,masks = encoding_func(pipe_emmdit, pr[idx], device)
                    mds_sample['text_feat'] = embs.cpu().detach().numpy().tobytes() #
                    mds_sample['text_mask'] = masks.cpu().detach().numpy().tobytes() #
                    
                    
                writer.write(mds_sample)    


        elif args.imgdim==1024:
            images = pipe(pr, width=1024,height=1024, guidance_scale=0.,max_sequence_length=256,num_inference_steps=4).images

            image_tensors = []
            for img in images:
                img = img.convert('RGB')
                tensor1 = image_transform(img).to(device)
                image_tensors.append(tensor1)
            image_batch = torch.stack(image_tensors)
            with torch.no_grad():
                latents_1024 = vae.encode(image_batch.to(torch.float32)).latent.to(torch.float16)
                
            latents_1024 = latents_1024.detach().clone().cpu().numpy()
        
            mds_samples = []
            for cur_prompt, cur_latent in zip(pr, latents_1024):
                mds_sample = {
                    'caption': cur_prompt,
                    'latents_1024': cur_latent.tobytes(),
                    }
                        
                mds_samples.append(mds_sample)
                
            for sample in mds_samples:
                writer.write(sample)

    writer.finish()


def main():
    args = parse_arguments()
    write_df(args)


if __name__ == "__main__":
    main()
