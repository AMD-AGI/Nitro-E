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


#download all
python prepare/download.py --dataset jdb --datadir prepare/datadir/jdb --num_proc 16
python prepare/download.py --dataset sa1b --datadir prepare/datadir/sa1b --skip_existing --num_proc 16 --max_image_size 1500


#Convert dataset to mds format.
#journeyDB
python prepare/convert_total.py \
  --dataset_type jdb \
  --images_dir prepare/datadir/jdb/raw/train/imgs \
  --captions_jsonl prepare/datadir/jdb/raw/train/train_anno_realease_repath.jsonl \
  --local_mds_dir prepare/datadir/mds/512jdb \
  --num_proc 4
#sa1b
python prepare/convert_total.py \
  --dataset_type sa1b \
  --images_dir prepare/datadir/sa1b/raw \
  --captions_dir prepare/datadir/sa1b/captions \
  --local_mds_dir prepare/datadir/mds/512sa1b \
  --num_proc 4

#Precompute latents.  
#journeyDB
accelerate launch  --num_machines 1 --num_processes 2 --gpu_ids 2,3  prepare/precompute.py --datadir prepare/datadir/mds/512jdb --savedir prepare/datadir/mds/mds_train_512jdb --vae --image_resolutions 512 --save_src

#sa1b
accelerate launch  --num_machines 1 --num_processes 2 --gpu_ids 2,3 prepare/precompute.py \
    --datadir prepare/datadir/mds/512sa1b \
    --savedir prepare/datadir/mds/mds_train_512sa1b \
    --vae --image_resolutions 512 --save_src



#Generate synthetic data for 512px model train total
bash scripts/down_ucsc_parquet.sh 70
HIP_VISIBLE_DEVICES=1 python prepare/generate_fluxtotal.py --local_mds_dir prepare/datadir/mds/mds_train_512ucsc_fromflux --imgdim 512 --prompt_type ucsc --download_mode total
HIP_VISIBLE_DEVICES=1 python prepare/generate_fluxtotal.py --local_mds_dir prepare/datadir/mds/mds_train_512diffdb_fromflux --imgdim 512 --prompt_type diffusiondb --download_mode total
