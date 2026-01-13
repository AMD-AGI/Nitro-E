# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

import torch

def calc_diff_loss(model_pred, target, weighting=None, mask=None):
    real_mask = None
    if weighting is not None:
        if mask is not None:
            real_mask = weighting * mask
        else:
            real_mask = weighting
    else:
        if mask is not None:
            real_mask = mask
    diff_loss = (model_pred - target) ** 2
    diff_loss = torch.mean(diff_loss.reshape(target.shape[0], -1), 1)


    if real_mask is not None:
        diff_loss = (real_mask.float() * diff_loss).sum() / real_mask.sum()
    else:
        diff_loss = torch.mean()

    
    return diff_loss

def calc_proj_loss(zs, zs_tilde):

    for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
        z_tilde_norm = torch.nn.functional.normalize(z_tilde, dim=-1) 
        z_norm = torch.nn.functional.normalize(z, dim=-1) 
        proj_loss = -1 * (z_tilde_norm * z_norm).sum(dim=-1).mean(-1)
    proj_loss /= len(zs)

    return proj_loss.mean()
