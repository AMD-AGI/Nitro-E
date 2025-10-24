# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/cache
#step1
accelerate launch --config_file configs/accelerate_config.yaml train.py --config configs/emmdit_512train_step1.yaml
#step2
#accelerate launch --config_file configs/accelerate_config.yaml train.py --config configs/emmdit_512train_step2.yaml
