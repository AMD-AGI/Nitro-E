# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

import os
import shutil
import tarfile
import argparse
import subprocess
import numpy as np
import requests
import urllib.request
from tqdm import tqdm
from multiprocessing import Pool
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from typing import List
from urllib.error import HTTPError, URLError

#Reference https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/datasets/prepare/sa1b/download.py
already_extracted = []

def download_and_extract(
    file_name: str,
    url: str,
    args: argparse.Namespace
) -> None:
    tar_dir, images_dir = args.datadir_compressed, args.datadir_raw
    downsize = transforms.Resize(
        args.max_image_size,
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC
    )

    # Check if the file already exists
    if not os.path.exists(f'{tar_dir}/{file_name}') and file_name not in already_extracted:
        print(f'Downloading {file_name} from {url}...')
        response = requests.get(url, stream=True)
        with open(f'{tar_dir}/{file_name}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f'{file_name} already exists in {tar_dir}. Skipping downloading it.')

    # Extract the file if it's a .tar file
    
    if file_name.endswith('.tar') and file_name not in already_extracted:
        images_subdir = os.path.join(images_dir, os.path.splitext(file_name)[0])
        os.makedirs(images_subdir, exist_ok=True)
        

        # Check if the file has already been extracted
        if len(os.listdir(images_subdir)) > 0 and args.skip_existing:
            print(f'{file_name} has already been extracted. Skipping extraction.')
        else:
            print(f'Extracting {file_name}...')
            with tarfile.open(f'{tar_dir}/{file_name}') as tar:
                for member in tqdm(tar.getmembers()):
                    try:

                        if member.name.endswith(".jpg"):
                            tar.extract(member, path=images_dir)
                            # Downsample images
                            p = os.path.join(images_dir, member.name.strip('./'))
                            new_p = os.path.join(images_subdir, member.name.strip('./'))
                            img = Image.open(p)
                            w, h = img.size
                            if min(w, h) > args.max_image_size:
                                img = downsize(img)
                            if min(w, h) < args.min_image_size:
                                print(
                                    f'Skipping image with resolution ({h}, {w}) - '
                                    f'Since at least one side has resolution below {args.min_image_size}'
                                )
                                continue
                            img.save(new_p)
                            os.remove(p)
                    except Exception as e:
                        print('Exception occured: ', e)
                
                print(f'{file_name} extracted!')

            already_extracted.append(file_name)
        
        # Delete tar file after extraction
        os.remove(f'{tar_dir}/{file_name}')
    else:
        print(f'{file_name} is not a tar file. Skipping extraction.')
        
  
def run_job(args):   
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.datadir_compressed, exist_ok=True)
    os.makedirs(args.datadir_raw, exist_ok=True)
    
    print('Downloading Llava synthetic captions for sa1b dataset')
    cap_dir = os.path.join(args.datadir, 'captions')
    os.makedirs(cap_dir, exist_ok=True)
    
    subprocess.run([
        "wget",
        'https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M/resolve/main/SA1B_caption.tar.gz',
        "-O",
        os.path.join(args.datadir, 'SA1B_caption.tar.gz')
    ], check=True)
    subprocess.run([
        "tar",
        "-xzvf",
        os.path.join(args.datadir, 'SA1B_caption.tar.gz'),
        "-C",
        cap_dir
    ], check=True)
    
    try:
        url = ('https://scontent-sin2-2.xx.fbcdn.net/m1/v/t6/An8MNcSV8eixKBYJ2kyw6sfPh-J9U4tH2BV7uPzibNa0pu4uHi6fyXdlbADVO4nfvsWpTwR8B0usCARHTz33cBQNrC0kWZsD1MbBWjw.txt?_nc_gid=2a96_scRHzU8ECRAEE-S6g&_nc_oc=AdkQWm7aRoBD5mDzTpI9526zYyBHzXskS7Gm1H842D0tq6ftvp23oQSFlsT-2k24cb4&ccb=10-5&oh=00_AfYLCY6QI4smnpsIoxusDm7OXaoSr5NDN7mhVIqCWnvy3Q&oe=6901BA58&_nc_sid=0fdd51')
        with urllib.request.urlopen(url) as f:
            links = [link.decode('utf-8') for link in f.readlines()[1:]]
    except (HTTPError, URLError) as e:
        print(
            f"Url no valid. Exception: {e}. Please manually update the above urls to the file "
            "containing the urls of each *.tar split. Its link dynamically updates at SA1B dataset "
            "website, thus we can't provide an automated download option permanently. Dataset webpage "
            "for text file: https://ai.meta.com/datasets/segment-anything-downloads/"
        )
        return
    
    print(f'Downloading only {args.data_fraction * 100}% of SA1B dataset')    
    links = links[:int(len(links) * args.data_fraction)]
    
    # Download and extract the files in parallel
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(
            download_and_extract,
            [(*line.strip().split('\t'), args) for line in links]
        )

