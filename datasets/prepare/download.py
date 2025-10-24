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

import argparse
import numpy as np
import os


#Reference https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/datasets/prepare/diffdb/download.py
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download and process datasets.")
    parser.add_argument('--dataset', type=str, choices=['jdb', 'diffusiondb', 'sa1b'], required=True)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--max_image_size', type=int, default=1024)
    parser.add_argument('--min_image_size', type=int, default=256)
    parser.add_argument('--num_proc', type=int, default=8)

    return parser

def get_parser_for_dataset(dataset_name: str):
    parser = parse_arguments()

    if dataset_name == 'jdb':
        from jdb_downloader import run_job
        parser.add_argument(
            '--valid_ids',
            type=int,
            nargs='+',
            default=list(np.arange(200)),
            help='List of valid image IDs (default is 0 to 199).',
        )
    
    elif dataset_name == 'diffusiondb':
        from diffdb_downloader import run_job
        parser.add_argument(
            '--valid_ids',
            type=int,
            nargs='+',
            default=list(np.arange(1, 14001)),
            help='List of valid image IDs (default is 1 to 14001).'
        )        

    elif dataset_name == 'sa1b':
        from sa1b_downloader import run_job
        parser.add_argument(
            '--data_fraction',
            type=float,
            default=1.0,
            help='Fraction of total dataset to download.'
        )
        parser.add_argument(
            '--skip_existing',
            action='store_true',
            help='Skip extraction if the file has already been extracted'
        )        

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return parser, run_job


def main():
    base_parser = parse_arguments()
    args_partial, _ = base_parser.parse_known_args()
    parser, run_job = get_parser_for_dataset(args_partial.dataset)
    args = parser.parse_args()

    args.datadir_compressed = os.path.join(args.datadir, 'compressed')
    args.datadir_raw = os.path.join(args.datadir, 'raw')

    run_job(args)
    

if __name__ == '__main__':
    main()
