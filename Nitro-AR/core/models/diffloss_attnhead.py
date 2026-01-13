# Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
from torch.utils.checkpoint import checkpoint
import math

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3)

from core.loss import calc_diff_loss

from core.utils import get_sigmas, retrieve_timesteps
from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed
from timm.models.vision_transformer import DropPath
from timm.layers import SwiGLU
import torch.nn.functional as F


from accelerate.logging import get_logger
logger = get_logger(__name__)

def cropped_pos_embed(pos_embed, height, width, patch_size=1, pos_embed_max_size=96):
    """Crops positional embeddings for SD3 compatibility."""
    if pos_embed_max_size is None:
        raise ValueError("`pos_embed_max_size` must be set for cropping.")

    height = height // patch_size
    width = width // patch_size
    if height > pos_embed_max_size:
        raise ValueError(
            f"Height ({height}) cannot be greater than `pos_embed_max_size`: {pos_embed_max_size}."
        )
    if width > pos_embed_max_size:
        raise ValueError(
            f"Width ({width}) cannot be greater than `pos_embed_max_size`: {pos_embed_max_size}."
        )

    top = (pos_embed_max_size - height) // 2
    left = (pos_embed_max_size - width) // 2
    spatial_pos_embed = pos_embed.reshape(1, pos_embed_max_size, pos_embed_max_size, -1)
    spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
    return spatial_pos_embed


# Source: https://github.com/NVlabs/Sana/blob/70459f414474c10c509e8b58f3f9442738f85577/diffusion/model/norms.py#L183
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
            Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return (self.weight * self._norm(x.float())).type_as(x)


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, 
                    shift,
                    grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleTransformerAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth
        )

        self.num_sampling_steps = num_sampling_steps

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_sampling_steps,
                                                    shift=shift)

    def forward(self, target, z, mask=None, masked_token_w=0):
        
        bs = target.shape[0]

        noise = torch.randn_like(target).to(z.dtype)

        # timestep sampling and mix latent with noise
        u = compute_density_for_timestep_sampling(
                weighting_scheme='logit_normal',
                batch_size=bs,
                logit_mean=0,
                logit_std=1,
                mode_scale=1.29,
            )

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        sigmas = get_sigmas(timesteps, self.noise_scheduler, n_dim=target.ndim).to(device=z.device)
        timesteps = timesteps.to(device=z.device)
        noisy_target = (1.0 - sigmas) * target + sigmas * noise

        noisy_target = noisy_target.to(target.dtype)

        model_pred = self.net(x=noisy_target,
                                                t=timesteps,
                                                c=z)
        
        diff_target = noise - target

        pred_x0 = noisy_target - sigmas * model_pred

        diff_loss = (model_pred - diff_target) ** 2

        if mask is None:
            diff_loss = diff_loss.mean()
            return diff_loss, pred_x0
        else:
            
            if masked_token_w > 0:
                weighting = mask.float().clone()
                weighting[mask==0] = masked_token_w
                diff_loss = diff_loss.mean(-1) * weighting
                diff_loss = diff_loss.sum(1) / weighting.sum(1)
            else:
                diff_loss = diff_loss.mean(-1) * mask
                diff_loss = diff_loss.sum(1) / mask.sum(1)

            diff_loss = diff_loss.mean()


            return diff_loss, pred_x0

    def sample(self, z, temperature=1.0, cfg=1.0, num_inference_steps=20):

        if not cfg == 1.0:
            latents = torch.randn(z.shape[0]//2, z.shape[1], self.in_channels).cuda().to(z.dtype)
        else:
            latents = torch.randn(z.shape[0], z.shape[1], self.in_channels).cuda().to(z.dtype)


        if num_inference_steps == 1:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.noise_scheduler, num_inference_steps)
        else: # num_inference_steps == 2:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.noise_scheduler, num_inference_steps+1)
            timesteps = [t.long().float() for t in timesteps if t != 1]

        for i, t in enumerate(timesteps):
            current_timestep = t
            
            if cfg != 1.0:
                latents_input = torch.cat([latents, latents], dim=0)

                current_timestep = current_timestep.expand(latents_input.shape[0]).cuda()
                output = self.net.forward(x=latents_input, t=current_timestep,
                                c=z)
                noise_pred_uncond, noise_pred_text = output.chunk(2)
                output = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

                latents = self.noise_scheduler.step(output, t, latents, return_dict=False)[0].to(z.dtype)
            else:
                latents_input = latents
                current_timestep = current_timestep.expand(latents_input.shape[0]).cuda()
                output = self.net.forward(x=latents_input, t=current_timestep,
                                c=z)
                latents = self.noise_scheduler.step(output, t, latents, return_dict=False)[0].to(z.dtype)

        return latents



class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        half_head_dim = dim // num_heads // 2
        hw_seq_len = 16

        if qk_norm:
            self.q_norm = RMSNorm(dim // num_heads)
            self.k_norm = RMSNorm(dim // num_heads)
        self.qk_norm = qk_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q,k,v).permute(0,2,1,3).reshape(B,N,C)
        x = self.proj(out)
        x = self.proj_drop(x)
        return x




class Block_v1(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            mlp_layer: nn.Module = SwiGLU,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio*2/3.),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), 
                                    nn.Linear(dim, 6*dim))

        self.dim=dim

    def forward(self, x: torch.Tensor, c) -> torch.Tensor:
        B,N,C = c.shape
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(C).view(B, N, 6, self.dim).unbind(2)


        x = x + self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1)).mul_(gamma1)
        x = x + self.mlp(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2)
        return x



def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(model_channels)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(C).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleTransformerAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        self.t_norm = RMSNorm(model_channels)
        self.c_norm = RMSNorm(model_channels)


        pos_embed_lv0 = get_2d_sincos_pos_embed(
                model_channels, 96, base_size=16, 
                interpolation_scale=1, output_type='pt'
            ) # [grid_size**2, embed_dim]
        pos_embed_lv0 = cropped_pos_embed(pos_embed_lv0,
                                                16, 
                                                16, 
                                                patch_size=1, pos_embed_max_size=96)
        pos_embed_lv0 = pos_embed_lv0.reshape(1, -1, pos_embed_lv0.shape[-1])
        self.register_buffer("pos_embed_lv0", pos_embed_lv0.float(), persistent=False)


        res_blocks = []

        for i in range(num_res_blocks):
            res_blocks.append(Block_v1(
            model_channels,model_channels//64,
            qk_norm=False
        ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        # t = (t*1000).long()
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.input_proj(x)


        if x.shape[1] > 1:
            x = x + self.pos_embed_lv0

        t = self.time_embed(t, x.dtype).unsqueeze(1)
        c = self.cond_embed(C)

        c = self.t_norm(t) + self.c_norm(C)

        for blk_ind, block in enumerate(self.res_blocks):
            x = block(x, c)

        return self.final_layer(x, c)


