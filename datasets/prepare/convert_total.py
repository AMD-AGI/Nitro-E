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
import json
from glob import glob
from argparse import ArgumentParser
from multiprocessing import Pool, current_process

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from streaming.base import MDSWriter
from streaming.base.util import merge_index


#Reference https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/datasets/prepare/jdb/convert.py
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["jdb", "sa1b", "diffusiondb"],
                        help="Dataset type to convert")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to local dir with images")
    parser.add_argument("--captions_jsonl", type=str, default=None,
                        help="Path to jsonl (for jdb)")
    parser.add_argument("--pqpath", type=str, default=None,
                        help="Path to parquet file (for diffusiondb)")
    parser.add_argument("--captions_dir", type=str, default=None,
                        help="Dir with txt captions (for sa1b)")
    parser.add_argument("--local_mds_dir", type=str, required=True,
                        help="Output dir for mds shards")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--safety_threshold", type=float, default=0.2,
                        help="For diffusiondb: discard if nsfw score > threshold")
    return parser.parse_args()


def current_process_index() -> int:
    p = current_process()
    return (p._identity[0] - 1) if p._identity else 0


columns = {
            #"width": "int32",
            #"height": "int32",
            "jpg": "jpeg",
            "caption": "str",
} 


def write_journeydb(args, lines, idx):
    proc_idx = current_process_index()
    save_dir = os.path.join(args.local_mds_dir, str(proc_idx))
    os.makedirs(save_dir, exist_ok=True)

    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    valid_archives = [os.path.basename(p) for p in glob(os.path.join(args.images_dir, "*"))]

    for f in tqdm(lines):
        d = json.loads(f)
        cap, p = d["prompt"], d["img_path"].strip("./")
        if os.path.dirname(p) not in valid_archives:
            continue
        try:
            img = Image.open(os.path.join(args.images_dir, p))
            w, h = img.size
            mds_sample = {"jpg": img, "caption": cap, }
            writer.write(mds_sample)
        except Exception as e:
            print(f"Skip sample, error {e}")
    writer.finish()


def write_diffusiondb(args, df, idx):
    proc_idx = current_process_index()
    save_dir = os.path.join(args.local_mds_dir, str(proc_idx))


    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    total, skipped = 0, 0
    for id, im, cap, s1, s2 in tqdm(
        zip(df["part_id"], df["image_name"], df["prompt"],df["image_nsfw"], df["prompt_nsfw"]) ):
        #if s1 > args.safety_threshold or s2 > args.safety_threshold:
        #    skipped += 1
        #    continue
        try:
            
            img_path = os.path.join(args.images_dir, f"part-{id:>06}/{im}")
            if not os.path.exists(img_path):
               
                continue
            img = Image.open(img_path)
            w, h = img.size
            mds_sample = {"jpg": img, "caption": cap, }
            writer.write(mds_sample)
            total += 1
        except Exception as e:
            print(f"Skip sample, error {e}")
    print(f"Written: {total}, Skipped: {skipped}")
    writer.finish()

def write_sa1b(images, args):
    proc_idx = current_process_index()
    save_dir = os.path.join(args.local_mds_dir, str(proc_idx))
    os.makedirs(save_dir, exist_ok=True)

    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for f in tqdm(images):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(f)
                w, h = img.size
                cap_path = os.path.join(args.captions_dir,
                                        os.path.basename(f).split(".")[0] + ".txt")
                cap = open(cap_path, "r").read().strip()
                mds_sample = {"jpg": img, "caption": cap,}
                writer.write(mds_sample)
            except Exception as e:
                print(f"Skip sample, error {e}")
    writer.finish()


def main():
    args = parse_arguments()
    

    if args.dataset_type == "jdb":
        metadata = list(open(args.captions_jsonl, "r"))
        metadata = np.array_split(metadata, args.num_proc)
        with Pool(args.num_proc) as pool:
            pool.starmap(write_journeydb, [(args, m, i) for i, m in enumerate(metadata)])

    elif args.dataset_type == "sa1b":
        images = glob(os.path.join(args.images_dir, "**", "*jpg"))
        images = np.array_split(images, args.num_proc)
        with Pool(args.num_proc) as pool:
            pool.starmap(write_sa1b, [(im, args) for im in images])

    elif args.dataset_type == "diffusiondb":
        metadata = pd.read_parquet(args.pqpath, engine="fastparquet")
        metadata = np.array_split(metadata, args.num_proc)
        with Pool(args.num_proc) as pool:
            pool.starmap(write_diffusiondb, [(args, df, i) for i, df in enumerate(metadata)])

    # merge shard indexes
    shards_metadata = [
        os.path.join(args.local_mds_dir, str(i), "index.json")
        for i in range(args.num_proc)
    ]
    merge_index(shards_metadata, out=args.local_mds_dir, keep_local=True)


if __name__ == "__main__":
    main()
