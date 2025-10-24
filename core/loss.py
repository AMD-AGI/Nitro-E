# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

import torch

def calc_diff_loss(model_pred, target, weighting=None):
    if weighting is not None:
        diff_loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
    else:
        diff_loss = torch.mean(
                        ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
    return diff_loss.mean()

def calc_proj_loss(zs, zs_tilde, ids_keep=None):
    
    bsz = zs[0].shape[0]
    #if ids_keep is not None:
    #    for zs_ind in range(len(zs)):
    #        zs[zs_ind] = torch.gather(zs[zs_ind], dim=1, index=ids_keep.reshape(bsz, -1, 1).repeat(1, 1, zs[zs_ind].shape[2]))

    for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
        z_tilde_norm = torch.nn.functional.normalize(z_tilde, dim=-1) 
        z_norm = torch.nn.functional.normalize(z, dim=-1) 
        proj_loss = -1 * (z_tilde_norm * z_norm).sum(dim=-1).mean(-1)
    proj_loss /= len(zs)

    return proj_loss.mean()
