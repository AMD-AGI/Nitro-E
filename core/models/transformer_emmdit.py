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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin
from diffusers.configuration_utils import register_to_config,ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import is_torch_version, logging
from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormSingle


from diffusers.models.attention_processor import Attention, AttentionProcessor,  JointAttnProcessor2_0
from diffusers.models.normalization import SD35AdaLayerNormZeroX
from diffusers.models.attention import FeedForward, _chunked_feed_forward

from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


#Reference https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py#L524
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


#Reference https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L570
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
        subsample_ratio = 1,
        subsample_seq_len = 1,
    ):
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_single"

        self.norm1 = nn.LayerNorm(dim)
        
        assert subsample_ratio >= 1 and subsample_seq_len >= 1
        self.subsample_ratio = subsample_ratio
        self.subsample_seq_len = subsample_seq_len

        self.norm1_context = nn.LayerNorm(dim)

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

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", mult=3)

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", mult=3)
        else:
            self.norm2_context = None
            self.ff_context = None

        #AdaLN-affine 
        self.scale_shift_bias = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        self.scale_shift_scale = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        

        if not context_pre_only:
            self.scale_shift_bias_c = nn.Parameter(torch.randn(6, dim) / dim**0.5)
            self.scale_shift_scale_c = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        embedded_timestep: torch.FloatTensor = None,
    ):
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        batch_size = hidden_states.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_bias[None] + temb.reshape(batch_size, 6, -1) * ( 1 + self.scale_shift_scale[None])
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        else:
            batch_size = hidden_states.shape[0]
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
                self.scale_shift_bias_c[None] + temb.reshape(batch_size, 6, -1) * (1 + self.scale_shift_scale_c)
            ).chunk(6, dim=1)
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_msa) + c_shift_msa

        #Alternating Subregion Attention (ASA)
        #print(self.subsample_ratio)
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

        #Alternating Subregion Attention (ASA)
        if self.subsample_ratio > 1:
            attn_output = rearrange(attn_output, 
                '(b s) (l n) c -> b (l s n) c', 
                n=self.subsample_seq_len, s=self.subsample_ratio)
            context_attn_output = rearrange(context_attn_output, 
                '(b s) (l n) c -> b (l s n) c', 
                n=self.subsample_seq_len, s=self.subsample_ratio)


        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output


        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

        return encoder_hidden_states, hidden_states


class TokenCompressor(nn.Module):
    def __init__(self, n_feat, ratio2x=2, ratio4x=4):
        super(TokenCompressor, self).__init__()
        self.body2x = nn.Sequential(
                                  nn.PixelUnshuffle(ratio2x),
                                  nn.Conv2d(n_feat*(ratio2x*ratio2x), n_feat, kernel_size=1, stride=1, padding=0, bias=True),
                                  torch.nn.GELU('tanh'),
                                  nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.body4x = nn.Sequential(
                                  nn.PixelUnshuffle(ratio4x),
                                  nn.Conv2d(n_feat*(ratio4x*ratio4x), n_feat, kernel_size=1, stride=1, padding=0, bias=True),
                                  torch.nn.GELU('tanh'),
                                  nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=True))
        
    def forward(self, x, cur_height, cur_width):
           
        hidden_states_2x = self.body2x(rearrange(x, "n (h w) c -> n c h w", h=cur_height, w=cur_width))
        hidden_states_2x = rearrange(hidden_states_2x, "n c h w  -> n (h w) c", h=int(cur_height / 2), w=int(cur_width / 2))

        hidden_states_4x = self.body4x(rearrange(x, "n (h w) c -> n c h w", h=cur_height, w=cur_width))
        hidden_states_4x = rearrange(hidden_states_4x, "n c h w  -> n (h w) c", h=int(cur_height / 4), w=int(cur_width / 4))

        x = torch.concat([hidden_states_2x, hidden_states_4x], dim=1)
        
        return x
                
    
class TokenReconstructor(nn.Module):
    def __init__(self, n_feat, ratio2x=2, ratio4x=4):
        super(TokenReconstructor, self).__init__()
        self.body2x = nn.Sequential(nn.PixelShuffle(ratio2x),
                                  nn.Conv2d(n_feat//(ratio2x*ratio2x), n_feat, kernel_size=1, stride=1, padding=0, bias=True),
                                  torch.nn.GELU('tanh'),
                                  nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=True))
        self.body4x = nn.Sequential(nn.PixelShuffle(ratio4x),
                                  nn.Conv2d(n_feat//(ratio4x*ratio4x), n_feat, kernel_size=1, stride=1, padding=0, bias=True),
                                  torch.nn.GELU('tanh'),
                                  nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=True))
        self.mergers = nn.Sequential(
                                  nn.Linear(n_feat*3, n_feat),
                                  torch.nn.GELU('tanh'),
                                  nn.Linear(n_feat, n_feat))
        
    def forward(self, x_2x, x_4x, cur_hidden_states, cur_height, cur_width):
        
        x_2x = self.body2x(rearrange(x_2x, "n (h w) c -> n c h w", h=int(cur_height / 2), w=int(cur_width / 2)))
        x_2x = rearrange(x_2x, "n c h w  -> n (h w) c", h=cur_height, w=cur_width)

        x_4x = self.body4x(rearrange(x_4x, "n (h w) c -> n c h w", h=int(cur_height / 4), w=int(cur_width / 4)))
        x_4x = rearrange(x_4x, "n c h w  -> n (h w) c", h=cur_height, w=cur_width)

        hidden_states = torch.cat([x_2x, x_4x, cur_hidden_states], dim=2)
        hidden_states = self.mergers(hidden_states)
        
        return hidden_states
                
                      
#copy from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_sd3.py    
class EMMDiTTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):       
    @register_to_config
    def __init__(
        self,
        sample_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 32,
        num_layers: int = 24,
        attention_head_dim: int = 32,
        num_attention_heads: int = 24,
        caption_channels: int = 2048,
        caption_projection_dim: int = 768,
        out_channels: int = 32,
        interpolation_scale: int = None,
        pos_embed_max_size: int = 96,
        use_sub_attn: bool=True, 
        qk_norm: str = 'rms_norm',
        repa_depth = -1,
        projector_dim=2048,
        z_dims=[768],
    ):
        super().__init__()
        self.block_groups = nn.ModuleList()
        self.tokencompressor = nn.ModuleList()
        self.tokenreconstructor = nn.ModuleList()
     

        self.time_text_embed = None
        self.patch_mixer_depth = None # initially no masking applied
        self.mask_ratio = 0
        self.use_sub_attn = use_sub_attn # set if use ASA 

        self.block_split_stage = [4, 16, 4]
        
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

        interpolation_scale = max(self.config.sample_size // 16, 1)

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
            pos_embed_max_size=pos_embed_max_size, 
        )

        pos_embed_lv0 = get_2d_sincos_pos_embed(
                self.inner_dim, pos_embed_max_size, base_size=self.config.sample_size // self.config.patch_size, 
                interpolation_scale=interpolation_scale, output_type='pt'
            ) 
        
        pos_embed_lv0 = cropped_pos_embed(pos_embed_lv0,
                                                self.config.sample_size, 
                                                self.config.sample_size, 
                                                patch_size=1, pos_embed_max_size=pos_embed_max_size)

        pos_embed_lv0 = pos_embed_lv0.reshape(1, -1, pos_embed_lv0.shape[-1])      

        self.register_buffer("pos_embed_lv0", pos_embed_lv0.float(), persistent=False)

        self.context_embedder = nn.Linear(self.config.caption_channels, self.config.caption_projection_dim)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=False
        )

        self.transformer_blocks = None

        if self.use_sub_attn == True:
            subample_ratio_list = [1, 4, 4]
            seq_len_list = [1, 1, 4]
        else:
            subample_ratio_list = [1, 1, 1]
            seq_len_list = [1, 1, 1]
        cur_ind = 0

        
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
                    
                    subsample_ratio=subample_ratio_list[cur_ind%len(subample_ratio_list)],
                    subsample_seq_len=seq_len_list[cur_ind%len(seq_len_list)],
                ))
                cur_ind += 1

            cur_group = nn.ModuleList(cur_group)

            self.block_groups.append(cur_group)

        ds_num = int(len(self.block_split_stage) // 2)
       

        for _ in range(ds_num):
            self.tokencompressor.append(nn.ModuleList([TokenCompressor(self.inner_dim, 2, 4)]))
       
        for _ in range(ds_num):
            self.tokenreconstructor.append(nn.ModuleList([TokenReconstructor(self.inner_dim, 2, 4)])) 


        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
    
            
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
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

        height, width = hidden_states.shape[-2:]

        cur_height = height // self.config.patch_size
        cur_width = width // self.config.patch_size

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.

        temb, embedded_timestep = self.adaln_single(
            timestep, None, batch_size=hidden_states.shape[0], hidden_dtype=hidden_states.dtype
        )

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

       
        len_keep = hidden_states.shape[1]
        zs = None


        ds_num = int(len(self.block_split_stage) // 2)
        encoder_feats = []
        for grp_ids, blocks in enumerate(self.block_groups):
            # for encoders
            for index_block, block in enumerate(blocks):
                # Skip specified layers
                is_skip = True if skip_layers is not None and index_block in skip_layers else False

                if torch.is_grad_enabled() and not is_skip:

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
                        temb,
                        joint_attention_kwargs,
                        **ckpt_kwargs,
                    )
                elif not is_skip:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                
                
            if self.repa_depth != -1 and grp_ids == 0:
                if self.training:
                    zs = [projector(hidden_states) for projector in self.projectors]
            if grp_ids < ds_num:
                encoder_feats.append(hidden_states)

                hidden_states = self.tokencompressor[grp_ids][0](hidden_states, cur_height, cur_width)                
                
            elif grp_ids < len(self.block_split_stage)-1:
                x_2x, x_4x = torch.split(hidden_states, [int((cur_height / 2)**2), int((cur_height / 4)**2)], dim=1)
                
                hidden_states = self.tokenreconstructor[grp_ids-ds_num][0](x_2x, x_4x, encoder_feats[len(encoder_feats)-1-(grp_ids-ds_num)], cur_height, cur_width)
                      
                hidden_states = hidden_states + self.pos_embed_lv0 #Position Reinforcement 

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if not self.training:
            # unpatchify
            patch_size = self.config.patch_size
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
            )


            if not return_dict:
                return (output,)

            return Transformer2DModelOutput(sample=output)
        
        else:
            return hidden_states, zs

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_flash_attn_processor(self):

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module):
            if hasattr(module, "set_processor"):
                if name.endswith('attn1'):
                    module.set_processor(AttnProcessor2_0_SAFA())

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module)
