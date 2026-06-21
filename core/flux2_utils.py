# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: [MIT]
#
# Flux2 latent utilities. Logic copied from Hugging Face diffusers:
# - pipeline_flux2_klein.py (Flux2KleinPipeline)
# - train_dreambooth_lora_flux2_klein.py
#
# Flux2 VAE format: encode -> (B, 32, H, W) -> patchify 2x2 -> (B, 128, H/2, W/2) -> BN normalize

from typing import Optional

import torch

__all__ = [
    "retrieve_latents",
    "patchify_latents",
    "unpatchify_latents",
    "pack_latents",
    "unpack_latents_with_ids",
    "prepare_latent_ids",
    "prepare_text_ids",
    "encode_vae_image",
    "decode_latents",
]


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "argmax",
) -> torch.Tensor:
    """Extract latents from VAE encoder output. Copied from diffusers pipeline utils."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    2x2 patchify: (B, C, H, W) -> (B, C*4, H/2, W/2).
    Copied from Flux2KleinPipeline._patchify_latents.
    """
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Inverse of patchify: (B, C*4, H, W) -> (B, C, H*2, W*2).
    Copied from Flux2KleinPipeline._unpatchify_latents.
    """
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), 2, 2, height, width
    )
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), height * 2, width * 2
    )
    return latents


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Pack spatial latent to sequence: (B, C, H, W) -> (B, H*W, C).
    Copied from Flux2KleinPipeline._pack_latents.
    """
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(
        0, 2, 1
    )
    return latents


def unpack_latents_with_ids(
    x: torch.Tensor, x_ids: torch.Tensor
) -> torch.Tensor:
    """
    Scatter packed tokens back to spatial using position ids.
    Copied from Flux2KleinPipeline._unpack_latents_with_ids.
    Returns: (B, C, H, W)
    """
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64, device=data.device)
        w_ids = pos[:, 2].to(torch.int64, device=data.device)
        h = torch.max(h_ids).item() + 1
        w = torch.max(w_ids).item() + 1
        flat_ids = (h_ids * w + w_ids).to(data.device)
        out = torch.zeros(
            (h * w, ch), device=data.device, dtype=data.dtype
        )
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """
    Generate 4D position coordinates (T, H, W, L) for latent tensors.
    Copied from Flux2KleinPipeline._prepare_latent_ids.
    Returns: (B, H*W, 4)
    """
    batch_size, _, height, width = latents.shape
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids


def prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    """
    Generate position ids for text embeddings (B, L, D).
    Copied from Flux2KleinPipeline._prepare_text_ids.
    """
    B, L, _ = x.shape
    t = torch.arange(1)
    h = torch.arange(1)
    w = torch.arange(1)
    l_idx = torch.arange(L)
    coords = torch.cartesian_prod(t, h, w, l_idx)
    return coords.unsqueeze(0).expand(B, -1, -1)


def encode_vae_image(
    vae,
    image: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "argmax",
) -> torch.Tensor:
    """
    Encode image with Flux2 VAE: encode -> patchify -> BN normalize.
    Copied from Flux2KleinPipeline._encode_vae_image.
    Requires AutoencoderKLFlux2 with .bn (batch norm) and config.batch_norm_eps.
    Returns: (B, 128, H/2, W/2) for 512px input -> (B, 128, 32, 32)
    """
    if image.ndim != 4:
        raise ValueError(f"Expected image dims 4, got {image.ndim}.")
    encoder_output = vae.encode(image)
    image_latents = retrieve_latents(
        encoder_output, generator=generator, sample_mode=sample_mode
    )
    image_latents = patchify_latents(image_latents)
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        image_latents.device, image_latents.dtype
    )
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    )
    image_latents = (image_latents - latents_bn_mean) / latents_bn_std
    return image_latents


def decode_latents(vae, latents: torch.Tensor):
    """
    Decode Flux2 latents: reverse BN -> unpatchify -> vae.decode.
    For use when model outputs patchified latent space (B, 128, H, W).
    Requires AutoencoderKLFlux2 with .bn.
    Returns output of vae.decode (typically .sample attribute for image tensor).
    """
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    )
    latents = latents * latents_bn_std + latents_bn_mean
    latents = unpatchify_latents(latents)
    return vae.decode(latents)
