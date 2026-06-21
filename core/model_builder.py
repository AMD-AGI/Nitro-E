# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: [MIT]

import importlib
from omegaconf import OmegaConf


# Keys used by training script, not passed to transformer constructor
_MODEL_ONLY_KEYS = {"_target_", "flashSA", "caption_max_seq_length"}


def build_transformer(model_cfg):
    """Instantiate transformer from merged model config."""
    cfg = OmegaConf.to_container(model_cfg, resolve=True)
    target = cfg.pop("_target_", None)
    if target is None:
        raise ValueError("model must specify _target_: dotted path to transformer class")

    params = {k: v for k, v in cfg.items() if k not in _MODEL_ONLY_KEYS}

    module_path, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(**params)
