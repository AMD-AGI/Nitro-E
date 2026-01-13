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

from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version, logging
from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormSingle

from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.normalization import SD35AdaLayerNormZeroX
from diffusers.models.attention import FeedForward, _chunked_feed_forward

from core.models.diffloss_attnhead import DiffLoss, RMSNorm

from core.utils import random_masking, sample_orders, patchify, mask_by_order

import numpy as np

from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

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


class JointTransformerBlockSingleNorm(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://huggingface.co/papers/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        subsample_ratio = 1,
        subsample_seq_len = 1,
    ):
        super().__init__()

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_single"


        self.norm1 = RMSNorm(dim)
        
        assert subsample_ratio >= 1 and subsample_seq_len >= 1
        self.subsample_ratio = subsample_ratio
        self.subsample_seq_len = subsample_seq_len

        # self.norm1_context = nn.LayerNorm(dim)
        self.norm1_context = RMSNorm(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-6,
        )

        if use_dual_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=1e-6,
            )
        else:
            self.attn2 = None

        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", mult=3)

        if not context_pre_only:
            self.norm2_context = RMSNorm(dim)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", mult=3)
        else:
            self.norm2_context = None
            self.ff_context = None

        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        norm_hidden_states = self.norm1(hidden_states)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)

        if self.subsample_ratio > 1:
            norm_hidden_states = rearrange(norm_hidden_states, 
                                           'b (l s n) c -> (b s) (l n) c', 
                                           n=self.subsample_seq_len, s=self.subsample_ratio)
            norm_encoder_hidden_states = rearrange(norm_encoder_hidden_states, 
                                           'b (l s n) c -> (b s) (l n) c', 
                                           n=self.subsample_seq_len, s=self.subsample_ratio)

        
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )
        if self.subsample_ratio > 1:
            attn_output = rearrange(attn_output, 
                                           '(b s) (l n) c -> b (l s n) c', 
                                           n=self.subsample_seq_len, s=self.subsample_ratio)
            context_attn_output = rearrange(context_attn_output, 
                                           '(b s) (l n) c -> b (l s n) c', 
                                           n=self.subsample_seq_len, s=self.subsample_ratio)

        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            if self._chunk_size is not None:
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + context_ff_output

        return encoder_hidden_states, hidden_states


class Downsample(nn.Module):
    def __init__(self, n_feat, ratio=2):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
                                  nn.PixelUnshuffle(ratio),
                                  nn.Conv2d(n_feat*(ratio*ratio), n_feat, kernel_size=1, stride=1, padding=0, bias=True),
                                  torch.nn.GELU('tanh'),
                                  nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return self.body(x)
    
class Upsample(nn.Module):
    def __init__(self, n_feat, ratio=2):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.PixelShuffle(ratio),
                                  nn.Conv2d(n_feat//(ratio*ratio), n_feat, kernel_size=1, stride=1, padding=0, bias=True),
                                  torch.nn.GELU('tanh'),
                                  nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return self.body(x)
    
class MMDiTTransformer2DModel(SD3Transformer2DModel):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        out_channels (`int`, defaults to 16): Number of output channels.

    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 24,
        attention_head_dim: int = 32,
        num_attention_heads: int = 24,
        caption_channels: int = 4096,
        caption_projection_dim: int = 768,
        out_channels: int = 16,
        interpolation_scale: int = None,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
        repa_depth = -1,
        projector_dim=2048,
        z_dims=[768],
        diffloss_d=3,
        diffloss_w=1024,
        num_sampling_steps=1000,
        diffusion_batch_mul=4,
        shift=1,
        mask_ratio_min=0.7,
        ratio_sampling_method='mar',
        global_token_type='vae_thumbnail_gt',
        global_head='none',


    ):
        super().__init__(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            caption_projection_dim=caption_projection_dim,
            out_channels=out_channels,
            pos_embed_max_size=pos_embed_max_size,
            dual_attention_layers=dual_attention_layers,
            qk_norm=qk_norm,
        )

        self.time_text_embed = None

        self.patch_mixer_depth = None # initially no masking applied
        self.mask_ratio = 0

        self.block_split_stage = [4, 16, 4]

        self.ratio_sampling_method = ratio_sampling_method

        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        if repa_depth != -1:
            from core.models.projector import build_projector
            self.projectors = nn.ModuleList([
                build_projector(self.inner_dim, projector_dim, z_dim) for z_dim in z_dims
                ])
            
            assert repa_depth >= 0 and repa_depth < num_layers
            self.repa_depth = repa_depth



        self.global_token_type = global_token_type

        interpolation_scale = max(self.config.sample_size // 16, 1)

        self.x_proj = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)
        self.z_proj_ln = RMSNorm(self.inner_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.in_channels))

        pos_embed_lv0 = get_2d_sincos_pos_embed(
                self.inner_dim, pos_embed_max_size, base_size=self.config.sample_size // self.config.patch_size, 
                interpolation_scale=interpolation_scale, output_type='pt'
            ) # [grid_size**2, embed_dim]

        
        
        pos_embed_lv0 = cropped_pos_embed(pos_embed_lv0,
                                                self.config.sample_size, 
                                                self.config.sample_size, 
                                                patch_size=1, pos_embed_max_size=pos_embed_max_size)
        pos_embed_lv0 = pos_embed_lv0.reshape(1, -1, pos_embed_lv0.shape[-1])
        self.register_buffer("pos_embed_lv0", pos_embed_lv0.float(), persistent=False)

        self.context_embedder = nn.Linear(self.config.caption_channels, self.config.caption_projection_dim)

        self.transformer_blocks = None

        self.block_groups = nn.ModuleList()
        for grp_ids, cur_bks in enumerate(self.block_split_stage):
            cur_group = []
            for i in range(cur_bks):
                cur_group.append(JointTransformerBlockSingleNorm(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=(grp_ids==len(self.block_split_stage)-1) \
                                        and (i == cur_bks - 1),
                    qk_norm=qk_norm,
                    use_dual_attention=False,
                ))

            cur_group = nn.ModuleList(cur_group)
            self.block_groups.append(cur_group)

        ds_num = int(len(self.block_split_stage) // 2)
        self.downsamplers = nn.ModuleList()
        for _ in range(ds_num):
            self.downsamplers.append(nn.ModuleList([Downsample(self.inner_dim, 2), Downsample(self.inner_dim, 4)]))
        self.upsamplers = nn.ModuleList()
        for _ in range(ds_num):
            self.upsamplers.append(nn.ModuleList([Upsample(self.inner_dim, 2), Upsample(self.inner_dim, 4)])) 
        self.mergers = nn.ModuleList()
        for _ in range(ds_num):
            self.mergers.append(nn.Sequential(
                                  nn.Linear(self.inner_dim*3, self.inner_dim),
                                  torch.nn.GELU('tanh'),
                                  nn.Linear(self.inner_dim, self.inner_dim)))

        
        self.global_z_proj = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim),
                nn.SiLU(),
                nn.Linear(self.inner_dim, self.inner_dim),
            )


        self.gradient_checkpointing = False

        self.norm_out = None
        self.pos_embed = None
        self.proj_out = None

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.out_channels,
            z_channels=self.inner_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            shift=shift,
        )

        if global_head != 'none':
            self.diffloss_global = DiffLoss(
                target_channels=self.out_channels,
                z_channels=self.inner_dim,
                width=diffloss_w,
                depth=diffloss_d,
                num_sampling_steps=num_sampling_steps,
                shift=shift,
            )
        else:
            self.diffloss_global = None
        

        self.diffusion_batch_mul = diffusion_batch_mul

        self.mask_ratio_min = mask_ratio_min



    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward_transformer(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        # skip_layers: Optional[List[int]] = None,
        latent_width=None,
        latent_height=None,
        pred_global_token=False
    ) -> torch.FloatTensor:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        # height, width = hidden_states.shape[-2:]

        # cur_height = height // self.config.patch_size
        # cur_width = width // self.config.patch_size
        zs = None

        cur_height = latent_height
        cur_width = latent_width
        
        hidden_states = self.x_proj(hidden_states)
        if not pred_global_token:
            hidden_states = hidden_states + self.pos_embed_lv0

        hidden_states = self.z_proj_ln(hidden_states)

        # hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # cur_blk_ind = 0
        # tar_blk_ind = sum(self.block_split_stage) - 1 
        
        # if self.global_head != 'none':
        #     cur_blk_count = 0
        #     if self.global_depth != -1:
        #         assert self.global_depth <= tar_blk_ind
        #         tar_blk_ind = self.global_depth

        ds_num = int(len(self.block_split_stage) // 2)
        encoder_feats = []
        for grp_ids, blocks in enumerate(self.block_groups):
            # for encoders
            for index_block, block in enumerate(blocks):
                # Skip specified layers

                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        joint_attention_kwargs,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            if pred_global_token:
                hidden_states = self.global_z_proj(hidden_states)
                return hidden_states, None
                
            if grp_ids < ds_num:
                encoder_feats.append(hidden_states)

                hidden_states_2x = self.downsamplers[grp_ids][0](rearrange(hidden_states, "n (h w) c -> n c h w", h=cur_height, w=cur_width))
                hidden_states_2x = rearrange(hidden_states_2x, "n c h w  -> n (h w) c", h=int(cur_height / 2), w=int(cur_width / 2))

                hidden_states_4x = self.downsamplers[grp_ids][1](rearrange(hidden_states, "n (h w) c -> n c h w", h=cur_height, w=cur_width))
                hidden_states_4x = rearrange(hidden_states_4x, "n c h w  -> n (h w) c", h=int(cur_height / 4), w=int(cur_width / 4))

                hidden_states = torch.concat([hidden_states_2x, hidden_states_4x], dim=1)

            elif grp_ids < len(self.block_split_stage)-1:
                x_2x, x_4x = torch.split(hidden_states, [int((cur_height / 2)**2), int((cur_height / 4)**2)], dim=1)

                x_2x = self.upsamplers[grp_ids-ds_num][0](rearrange(x_2x, "n (h w) c -> n c h w", h=int(cur_height / 2), w=int(cur_width / 2)))
                x_2x = rearrange(x_2x, "n c h w  -> n (h w) c", h=cur_height, w=cur_width)

                x_4x = self.upsamplers[grp_ids-ds_num][1](rearrange(x_4x, "n (h w) c -> n c h w", h=int(cur_height / 4), w=int(cur_width / 4)))
                x_4x = rearrange(x_4x, "n c h w  -> n (h w) c", h=cur_height, w=cur_width)

                
                hidden_states = torch.cat([x_2x, x_4x, encoder_feats[len(encoder_feats)-1-(grp_ids-ds_num)]], dim=2)
                hidden_states = self.mergers[grp_ids-ds_num](hidden_states)
                hidden_states = hidden_states + self.pos_embed_lv0
            

        return hidden_states, zs


    # def forward_loss(self, z, target, mask):
    #     bsz, seq_len, _ = target.shape
    #     target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
    #     z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
    #     mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
    #     loss, pred_x0 = self.diffloss(z=z, target=target, mask=mask)
    #     return loss, pred_x0
    
    def forward(
        self,
        img_feat,
        text_emb,
        global_token=None,
        pred_global_token=False,
        global_token_dependency=True,
        masked_token_w=0):

        if pred_global_token:
            if 'vae_thumbnail' in self.global_token_type:
                img_patches = patchify(img_feat)
            elif 'dino' in self.global_token_type:
                img_patches = self.feat_adapt(img_feat)
            input_tensor = self.mask_token.repeat(img_patches.size(0), 1, 1)

            hidden_states, zs = self.forward_transformer(hidden_states=input_tensor,
                                            encoder_hidden_states=text_emb,
                                            pred_global_token=pred_global_token)

            if self.diffloss_global is not None:
                loss, pred_x0 = self.diffloss_global(z=hidden_states, target=img_patches)
            else:
                loss, pred_x0 = self.diffloss(z=hidden_states, target=img_patches)


            return loss, pred_x0, zs
        
        else:
            height, width = img_feat.shape[-2:]
            img_patches = patchify(img_feat)
            if self.global_token_type in ['vae_thumbnail_gt', 'vae_thumbnail_ncl']:
                global_token = patchify(global_token)
            elif 'dino' in self.global_token_type:
                global_token = self.feat_adapt(global_token)

            orders = sample_orders(bsz=img_patches.size(0), seq_len=height*width)
            mask, mask_rate = random_masking(img_patches, orders, self.mask_ratio_min, 
                                sampling_method=self.ratio_sampling_method)
            

            mask_expanded = mask.unsqueeze(-1)

            if global_token_dependency:
                masked_img_patches = torch.where(mask_expanded.bool(), global_token, img_patches)
            else:
                if mask_rate == 1:
                    masked_img_patches = torch.where(mask_expanded.bool(), global_token, img_patches)
                else:
                    masked_img_patches = torch.where(mask_expanded.bool(), self.mask_token, img_patches)


            hidden_states, zs = self.forward_transformer(hidden_states=masked_img_patches,
                                                encoder_hidden_states=text_emb,
                                                latent_height=height,
                                                latent_width=width)

            loss, pred_x0 = self.diffloss(z=hidden_states, target=img_patches, mask=mask, masked_token_w=masked_token_w)

            return loss, pred_x0, zs

    def sample_tokens(self, txt_emb, uncond_txt_emb, cfg=1.0, num_iter=1,
                        cfg_schedule="constant", temperature=1.0, progress=False, 
                        diff_steps=20, pred_global_token=False, global_token=None, rewrite=False,
                        global_token_dependency=True):

        if pred_global_token:
            bsz = txt_emb.shape[0]
            tokens = self.mask_token.repeat(bsz, 1, 1)

            if not cfg == 1.0:
                uncond_txt_emb = uncond_txt_emb.repeat(bsz, 1, 1)
                txt_emb = torch.cat([uncond_txt_emb, txt_emb], dim=0)
            

            # generate latents
            cur_tokens = tokens.clone()

            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                
            hidden_states, _ = self.forward_transformer(hidden_states=tokens,
                                            encoder_hidden_states=txt_emb,
                                            pred_global_token=True)



            # sampled_token_latent = self.diffloss.sample(hidden_states, temperature=temperature, 
                                                        # cfg=cfg, num_inference_steps=diff_steps)
            
            if self.diffloss_global is not None:
                sampled_token_latent = self.diffloss_global.sample(hidden_states, temperature=temperature, 
                                                        cfg=cfg, num_inference_steps=diff_steps)
            else:
                sampled_token_latent = self.diffloss.sample(hidden_states, temperature=temperature, 
                                                        cfg=cfg, num_inference_steps=diff_steps)

            return sampled_token_latent, _
        else:
            self.seq_len = 256
            bsz = txt_emb.shape[0]

            mask = torch.ones(bsz, self.seq_len).cuda().to(txt_emb.dtype)
            tokens = global_token.repeat(1, self.seq_len, 1)
            orders = sample_orders(bsz, self.seq_len)

            if not cfg == 1.0:
                uncond_txt_emb = uncond_txt_emb.repeat(bsz, 1, 1)
                txt_emb = torch.cat([uncond_txt_emb, txt_emb], dim=0)
            

            indices = list(range(num_iter))
            if progress:
                indices = tqdm(indices)
            
            inter_results = []

            # generate latents
            for step in indices:
                cur_tokens = tokens.clone()

                if not cfg == 1.0:
                    tokens = torch.cat([tokens, tokens], dim=0)
                    mask = torch.cat([mask, mask], dim=0)
                    
                hidden_states, _ = self.forward_transformer(hidden_states=tokens,
                                                encoder_hidden_states=txt_emb,
                                                latent_height=16,
                                                latent_width=16)

                # mask ratio for the next round, following MaskGIT and MAGE.
                mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
                
                mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

                # masks out at least one for the next iteration
                mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                        torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
                # get masking for next iteration and locations to be predicted in this iteration
                mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
                

                if rewrite:
                    if step >= num_iter - 1:
                        mask_to_pred = torch.ones_like( mask[:bsz], dtype=torch.bool)
                    else:
                        mask_to_pred = torch.logical_not(mask_next.bool())
                else:
                    if step >= num_iter - 1:
                        mask_to_pred = mask[:bsz].bool()
                    else:
                        mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
                

                mask = mask_next


                if cfg_schedule == "linear":
                    cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
                elif cfg_schedule == "constant":
                    cfg_iter = cfg
                else:
                    raise NotImplementedError
                sampled_token_latent = self.diffloss.sample(hidden_states, temperature=temperature, 
                                                            cfg=cfg_iter, num_inference_steps=diff_steps)

                inter_results.append(sampled_token_latent.clone())
                if not global_token_dependency and step == 0:
                    cur_tokens = torch.ones(bsz, self.seq_len, 32).cuda().to(txt_emb.dtype) * self.mask_token
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent[mask_to_pred.nonzero(as_tuple=True)]
                tokens = cur_tokens.clone()
                inter_results.append(cur_tokens.clone())

            return tokens, inter_results



    def enable_gradient_checkpointing(self):
        self.diffloss.net.grad_checkpointing = True
        self.gradient_checkpointing = True