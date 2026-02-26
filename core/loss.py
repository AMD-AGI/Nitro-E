# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: [MIT]

import importlib
import torch
from omegaconf import OmegaConf


class BaseLoss:
    """Base class for configurable losses."""

    def __call__(self, model_pred, target, **kwargs):
        raise NotImplementedError


class DiffLoss(BaseLoss):
    """L2 diffusion loss with optional timestep weighting."""

    def __call__(self, model_pred, target, weighting=None, **kwargs):
        if weighting is not None:
            diff_loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                    target.shape[0], -1
                ),
                1,
            )
        else:
            diff_loss = torch.mean(
                ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
        return diff_loss.mean()


def build_loss(loss_cfg):
    """Instantiate loss from config with _target_ and params."""
    cfg = OmegaConf.to_container(loss_cfg, resolve=True)
    target = cfg.pop("_target_", None)
    if target is None:
        raise ValueError("loss must specify _target_: dotted path to loss class")

    params = cfg.pop("params", {})

    module_path, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(**params)
