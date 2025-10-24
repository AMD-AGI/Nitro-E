# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

import torch

def build_opt(opt_type, opt_params={}):
    if opt_type.lower() in ['adamw']:
        opt_class = torch.optim.AdamW
        opt_kwargs = {'lr': 1e-4,
                    'weight_decay': 3e-2,
                    'eps': 1e-10}
        opt_kwargs.update(opt_params)
    else:
        raise Exception()

    return opt_class, opt_kwargs     
