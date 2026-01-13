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
import torch.nn as nn
import einops
import numpy as np
import scipy.stats as stats
from typing import Callable, List, Optional, Tuple, Union
import inspect

# get sigmas from noise_scheduler
def get_sigmas(timesteps, noise_scheduler, n_dim=4):
    sigmas = noise_scheduler.sigmas
    schedule_timesteps = noise_scheduler.timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices]
    sigma = sigma.view(-1, *([1]*(n_dim-1)))
    return sigma.clone()

def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def token_drop(y,
               uncond_prompt_embeds, 
               class_dropout_prob=0.1):
    """
    Drops labels to enable classifier-free guidance.
    """
    drop_ids = torch.rand(y.shape[0]).cuda() < class_dropout_prob
    y = torch.where(drop_ids[:, None, None], uncond_prompt_embeds, y)
    return y
    # y_mask = torch.where(drop_ids[:, None], uncond_prompt_attention_mask, y_mask)
    # return y, y_mask


def get_null_embed(npz_file, device):
    data = torch.load(npz_file)
    uncond_prompt_embeds = data['uncond_prompt_embeds'].to(device)
    uncond_prompt_attention_mask = data['uncond_prompt_attention_mask'].to(device)
    return uncond_prompt_embeds, uncond_prompt_attention_mask

def patchify(x, patch_size=1):
    bsz, c, h, w = x.shape
    p = patch_size
    h_, w_ = h // p, w // p

    x = x.reshape(bsz, c, h_, p, w_, p)
    x = torch.einsum('nchpwq->nhwcpq', x)
    x = x.reshape(bsz, h_ * w_, c * p ** 2)
    return x  # [n, l, d]

def unpatchify(self, x, patch_size, vae_embed_dim):
    bsz = x.shape[0]
    p = patch_size
    c = vae_embed_dim
    h_, w_ = self.seq_h, self.seq_w

    x = x.reshape(bsz, h_, w_, c, p, p)
    x = torch.einsum('nhwcpq->nchpwq', x)
    x = x.reshape(bsz, c, h_ * p, w_ * p)
    return x  # [n, c, h, w]

def sample_orders(bsz, seq_len):
    # generate a batch of random generation orders
    orders = []
    for _ in range(bsz):
        order = np.array(list(range(seq_len)))
        np.random.shuffle(order)
        orders.append(order)
    orders = torch.Tensor(np.array(orders)).cuda().long()
    return orders

def random_masking(x, orders, mask_ratio_min, sampling_method='mar'):
    # generate token mask
    bsz, seq_len, embed_dim = x.shape

    if sampling_method == 'mar':
        mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        mask_rate = mask_ratio_generator.rvs(1)[0]
    elif sampling_method == 'maskgit':
        mask_rate = np.cos(np.random.uniform() * np.pi / 2)
    num_masked_tokens = int(np.ceil(seq_len * mask_rate))
    mask = torch.zeros(bsz, seq_len, device=x.device)
    mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.ones(bsz, seq_len, device=x.device))
    return mask

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps