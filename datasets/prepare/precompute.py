# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

import os
import time
from argparse import ArgumentParser
import numpy as np
import torch
from accelerate import Accelerator
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm
from diffusers import AutoencoderDC
from transformers import AutoModelForCausalLM,AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
import timm

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from datasets.prepare.base import build_streaming_total_precompute_dataloader,

from core.models.emmdit_pipeline import EMMDiTPipeline

def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Local directory to store mds shards.',
    )
    parser.add_argument(
        '--savedir',
        type=str,
        default='',
        help='Remote path to upload MDS-formatted shards to.',
    )
    parser.add_argument(
        '--image_resolutions',
        type=int,
        nargs='+',
        default=[512,1024],
        help='List of image resolutions to use for processing.',
    )

    parser.add_argument(
        '--model_dtype',
        type=str,
        choices=('float16', 'bfloat16', 'float32'),
        default='bfloat16',
        help='Data type for the encoding models',
    )
    parser.add_argument(
        '--save_dtype',
        type=str,
        choices=('float16', 'float32'),
        default='float16',
        help='Data type to save the latents. mds not support bfloat16',
    )
    parser.add_argument(
        '--vae',
        default=False,
        action='store_true',
        help='If True, write VAE feature to mds for vision encoding.',
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
    parser.add_argument(
        '--save_src',
        default=False,
        action='store_true',
        help='If True, also save images',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size per device to use for encoding.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2024,
        help='Seed for random number generation.',
    )
    args = parser.parse_args()
    if isinstance(args.image_resolutions, int):
        args.image_resolutions = [args.image_resolutions]
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

def main(args):
    """Precompute image and text latents and store them in MDS format.

    By default, we only save the image latents for 256x256 and 512x512 image
    resolutions (using center crop).

    Note that the image latents will be scaled by the vae_scaling_factor.
    """
    accelerator = Accelerator()
    device = accelerator.device
    device_idx = int(accelerator.process_index)

    # Set random seeds
    torch.manual_seed(device_idx + args.seed)
    torch.cuda.manual_seed(device_idx + args.seed)
    np.random.seed(device_idx + args.seed)

    dataloader = build_streaming_total_precompute_dataloader(
        datadir=[args.datadir],
        batch_size=args.batch_size,
        resize_sizes=[args.image_resolutions,224],
        drop_last=False,
        shuffle=False,
        prefetch_factor=2,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    print(f'Device: {device_idx}, Dataloader sample count: {len(dataloader.dataset)}')


    vae = AutoencoderDC.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=torch.bfloat16).to(device).eval()
    vae = torch.compile(vae)


    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B/')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B/', torch_dtype=torch.bfloat16)
    text_encoder.to(device)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    from core.models.transformer_emmdit import EMMDiTTransformer
   
    if args.image_resolutions==[512]:
        transformer = EMMDiTTransformer(
                in_channels=32,
                out_channels=32,
                sample_size=16,
                patch_size=1,
                caption_channels=2048,
                qk_norm='rms_norm',
            )
    elif args.image_resolutions==[1024]:
        transformer = EMMDiTTransformer(
                in_channels=32,
                out_channels=32,
                sample_size=32,
                patch_size=1,
                caption_channels=2048,
                qk_norm='rms_norm',
            )        

    pipe = EMMDiTPipeline(tokenizer,
        text_encoder,
        vae,
        transformer,
        scheduler)


    dinoencoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
    del dinoencoder.head
    dinoencoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(dinoencoder.pos_embed.data, [16, 16],)
    dinoencoder.head = torch.nn.Identity()
    dinoencoder = dinoencoder.to(device).eval()



    columns = {
        'caption': 'str',
        #'latents_512': 'bytes',
        #'latents_1024': 'bytes',
        #'jpg': 'jpeg',
        #'dino_feat': 'bytes',#online
        #'text_feat': 'bytes',#online
        #'text_mask': 'bytes' #online
    }

    if args.image_resolutions==[512]:
        columns['jpg'] = 'jpeg'
    if args.vae and args.image_resolutions==[512]:
        columns['latents_512'] = 'bytes'
    elif args.vae and args.image_resolutions==[1024]:
        columns['latents_1024'] = 'bytes'
    if args.dino_model:
        columns['dino_feat'] = 'bytes'
    if args.text_encoder:
        columns['text_feat'] = 'bytes'
        columns['text_mask'] = 'bytes'


    remote_upload = os.path.join(args.savedir, str(accelerator.process_index))
    writer = MDSWriter(
        out=remote_upload,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )


    for batch in tqdm(dataloader):
        image_re = torch.stack(batch['image_re0']).to(device)
        if args.dino_model:
            image_dino = torch.stack(batch['image_re1']).to(device)

        with torch.no_grad():
            latents = vae.encode(image_re.to(torch.bfloat16)).latent.to(torch.float16)
            """
            image_decode = vae.decode(latents.to(torch.bfloat16) / vae.scaling_factor).sample
            image_decode = (image_decode.to(torch.float32).clamp(-1, 1) + 1) / 2.0  # [0,1]
            image_decode = image_decode.cpu().permute(0, 2, 3, 1).numpy()  # N,H,W,C
            from PIL import Image
            image_decode = Image.fromarray((image_decode[0] * 255).astype("uint8"))
            image_decode.save('de.jpg')
            print(batch['sample'][0]['caption'])
            """
            latents = latents.detach().cpu().numpy()
    

            # Write the batch to the MDS file
            for i in range(latents.shape[0]):
                mds_sample = {'caption': batch['sample'][i]['caption']}
                
                if args.text_encoder:
                    embs,masks = encoding_func(pipe, batch['sample'][i]['caption'], device)
                    mds_sample['text_feat'] = embs.cpu().numpy().tobytes()
                    mds_sample['text_mask'] = masks.cpu().numpy().tobytes()
                
                if args.vae and args.image_resolutions==[512]:
                    mds_sample['latents_512'] = latents[i].tobytes()
                elif args.vae and args.image_resolutions==[1024]:
                    mds_sample['latents_1024'] = latents[i].tobytes() 
                if args.save_src:
                    mds_sample['jpg'] = batch['sample'][i]['jpg']

                if args.dino_model:
                    with torch.no_grad():
                        z = dinoencoder.forward_features(image_dino)
                        z = z['x_norm_patchtokens']
                    z = z.cpu().half().numpy()
                    mds_sample['dino_feat'] = z.tobytes()

                writer.write(mds_sample)
                


    writer.finish()

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    print(f'Process {accelerator.process_index} finished')
    time.sleep(10)

    # Merge the mds shards created by each device (only do on main process)
    if accelerator.is_main_process:
        shards_metadata = [
            os.path.join(args.savedir, str(i), 'index.json')
            for i in range(accelerator.num_processes)
        ]
        merge_index(shards_metadata, out=args.savedir, keep_local=True)


if __name__ == '__main__':
    main(parse_args())
