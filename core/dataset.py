# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from streaming import Stream, StreamingDataset
from typing import List, Dict, Union, Optional
import os
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

class StreamingLatentsDataset(StreamingDataset):
    """Dataset class for loading precomputed latents from mds format.
    
    Args:
        streams: List of individual streams (in our case streams of individual datasets)
        shuffle: Whether to shuffle the dataset
        latent_size: Size of latents
        caption_max_seq_length: Context length of text-encoder
        caption_channels: Dimension of caption embeddings
        cap_drop_prob: Probability of using all zeros caption embedding (classifier-free guidance)
        batch_size: Batch size for streaming
    """

    def __init__(
        self,
        streams: Optional[List[Stream]] = None,
        shuffle: bool = False,
        latent_size: Optional[int] = None,
        caption_max_seq_length: Optional[int] = None,
        caption_channels: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
            cache_limit='10tb',
            shuffle_algo='py1s',
        )

        self.latent_size = latent_size
        self.caption_max_seq_length = caption_max_seq_length
        self.caption_channels = caption_channels


        self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, 
                                     IMAGENET_DEFAULT_STD),
            ])

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        sample = super().__getitem__(index)

        #print(sample.keys()) #"caption", "dino_feat", "jpg", "latents_512", "text_feat", "text_mask"
        
        # Flux2 format: VAE encode (32,H,W) -> patchify 2x2 -> (128,H/2,W/2)
        # 512px: (128, 32, 32), 1024px: (128, 64, 64)
        if 'latents_512' in sample:
            image_latents = torch.from_numpy(
                np.frombuffer(sample['latents_512'], dtype=np.float16)
                .copy()
            ).reshape(128, self.latent_size, self.latent_size).float()
        if 'latents_1024' in sample:
            image_latents = torch.from_numpy(
                np.frombuffer(sample['latents_1024'], dtype=np.float16)
                .copy()
            ).reshape(128, self.latent_size, self.latent_size).float()
        
        dino_feat = torch.zeros([256,768])
        seq_len = self.caption_max_seq_length or 128
        channels = self.caption_channels or 2048
        text_feat = torch.zeros([seq_len, channels])
        if 'dino_feat' in sample:
            dino_feat = torch.from_numpy(
                np.frombuffer(sample['dino_feat'], dtype=np.float16)
                .copy()
            ).reshape(256, 768).float()
        
        if 'text_feat' in sample:
            text_feat = torch.from_numpy(
                np.frombuffer(sample['text_feat'], dtype=np.float16)
                .copy()
            ).reshape(seq_len, channels).float()
        # No text_mask: training computes attention mask from caption when precompute_txt_emb
        text_mask = None

        jpg_tensor = torch.zeros([3, 224, 224])
        if 'jpg' in sample:
            jpg = sample['jpg']
            jpg_tensor = self.transform(jpg)

        prompt = sample.get('caption', '')

        return image_latents, prompt, jpg_tensor, text_feat, text_mask, dino_feat


def build_streaming_latents_dataloader(
    dataset_config,
    batch_size: int,
    latent_size: int = 32,
    caption_max_seq_length: int = 120,
    caption_channels: int = 1024,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """Creates a DataLoader for streaming latents dataset."""

    streams = [Stream(remote=None, local=d) for d in dataset_config.datasets]

    dataset = StreamingLatentsDataset(
        streams=streams,
        shuffle=shuffle,
        latent_size=latent_size,
        caption_max_seq_length=caption_max_seq_length,
        caption_channels=caption_channels,
        batch_size=batch_size, #shuffle_block_size=26214,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader


class DummyDataset(Dataset):
    def __init__(self, latent_size, caption_max_seq_length, caption_channels):
        self.latent_size = latent_size
        self.caption_max_seq_length = caption_max_seq_length
        self.caption_channels = caption_channels
        self.transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, 
                                     IMAGENET_DEFAULT_STD),
            ])
    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        img_latent = torch.randn((128, self.latent_size, self.latent_size))
        txt_emb = torch.randn((self.caption_max_seq_length, self.caption_channels))
        dino_feat = torch.randn((256, 768))
        jpg_tensor = torch.zeros([3, 224, 224])
        prompt = ""  # DummyDataset: training computes mask from prompt when needed
        return (img_latent, prompt, jpg_tensor, txt_emb, None, dino_feat)
