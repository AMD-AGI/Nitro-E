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
import argparse
import subprocess
from glob import iglob
from multiprocessing import Pool
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

#Reference https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/datasets/prepare/jdb/download.py
def download_and_process_metadata(args: argparse.Namespace):
    # Only using a single process for downloading metadata
    metadata_files = [
        ('data/train', 'train_anno.jsonl.tgz'),
        ('data/train', 'train_anno_realease_repath.jsonl.tgz'),
        ('data/valid', 'valid_anno_repath.jsonl.tgz'),
        ('data/test', 'test_questions.jsonl.tgz'),
        ('data/test', 'imgs.tgz'),
    ]

    for subfolder, filename in metadata_files:
        hf_hub_download(
            repo_id="JourneyDB/JourneyDB",
            repo_type="dataset",
            subfolder=subfolder,
            filename=filename,
            local_dir=args.datadir_compressed,
            local_dir_use_symlinks=False,
        )

    metadata_tars = [
        os.path.join(dir, fname) for (dir, fname) in metadata_files
    ]

    for tar_file in metadata_tars:
        subprocess.call(
            f'tar -xvzf {args.datadir_compressed}/{tar_file} '
            f'-C {args.datadir_compressed}/{os.path.dirname(tar_file)}',
            shell=True,
        )

    shutil.copy(
        f'{args.datadir_compressed}/data/train/train_anno_realease_repath.jsonl',
        f'{args.datadir_raw}/train/train_anno_realease_repath.jsonl',
    )
    shutil.copy(
        f'{args.datadir_compressed}/data/valid/valid_anno_repath.jsonl',
        f'{args.datadir_raw}/valid/valid_anno_repath.jsonl',
    )
    shutil.copy(
        f'{args.datadir_compressed}/data/test/test_questions.jsonl',
        f'{args.datadir_raw}/test/test_questions.jsonl',
    )
    shutil.move(
        f'{args.datadir_compressed}/data/test/imgs',
        f'{args.datadir_raw}/test/',
    )


def download_uncompress_resize(
    args: argparse.Namespace,
    split: str,
    idx: int,
):
    """Download, uncompress, and resize images for a given archive index."""
    assert split in ('train', 'valid')
    assert idx in args.valid_ids

    print(f"Downloading idx: {idx}")
    hf_hub_download(
        repo_id="JourneyDB/JourneyDB",
        repo_type="dataset",
        subfolder=f'data/{split}/imgs',
        filename=f'{idx:>03}.tgz',
        local_dir=args.datadir_compressed,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded idx: {idx}")

    print(f"Extracting idx: {idx}")
    subprocess.call(
        f'tar -xzf {args.datadir_compressed}/data/{split}/imgs/{idx:>03}.tgz '
        f'-C {args.datadir_compressed}/data/{split}/imgs/',
        shell=True,
    )
    print(f"Extracted idx: {idx}")

    print(f"Removing idx: {idx}")
    os.remove(f'{args.datadir_compressed}/data/{split}/imgs/{idx:>03}.tgz')
    print(f"Removed idx: {idx}")

    # add bicubic downsize
    downsize = transforms.Resize(
        args.max_image_size,
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC,
    )

    print(f"Downsizing idx: {idx}")
    os.makedirs(
        f'{args.datadir_raw}/{split}/imgs/{idx:>03}/',
        exist_ok=True,
    )
    for f in iglob(f'{args.datadir_compressed}/data/{split}/imgs/{idx:>03}/*'):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                img = Image.open(f)
                w, h = img.size
                if min(w, h) > args.max_image_size:
                    img = downsize(img)
                if min(w, h) < args.min_image_size:
                    print(
                        f'Skipping image with resolution ({h}, {w}) - '
                        f'Since at least one side has resolution below {args.min_image_size}'
                    )
                    continue
                img.save(
                    f'{args.datadir_raw}/{split}/imgs/{idx:>03}/{os.path.basename(f)}'
                )
                os.remove(f)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Error {e}, File: {f}")
    print(f'Downsized idx: {idx}')


def run_job(args):
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(args.datadir_compressed, exist_ok=True)
    os.makedirs(args.datadir_raw, exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, 'train', 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, 'valid', 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(args.datadir_raw, 'test'), exist_ok=True)

    download_and_process_metadata(args)

    # Prepare arguments for multiprocessing
    pool_args = [('train', i) for i in args.valid_ids] + [('valid', i) for i in args.valid_ids]

    # Use multiprocessing to download, uncompress, and resize images
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(download_uncompress_resize, [(args, split, idx) for split, idx in pool_args])
