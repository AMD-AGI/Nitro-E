# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier:  [MIT]

import argparse
import datetime
import os
from tqdm import tqdm
import logging
import torch
from copy import deepcopy
import wandb
import einops

from accelerate.logging import get_logger
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers import get_constant_schedule_with_warmup
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3)
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM,AutoTokenizer

from core.utils import (token_drop,  
                        get_sigmas, patchify_and_apply_mask,
                        ema_update)
from core.opt import build_opt
from core.loss import calc_diff_loss, calc_proj_loss

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def main(args):
    default_cfg = OmegaConf.load("configs/default_config.yaml")
    custom_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, custom_cfg)
    

    proj_dir = os.path.join(cfg.work_root, cfg.exp_name)
    
    #init accelerator
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with='wandb',
        kwargs_handlers=[init_handler],
    )
    logger.info(accelerator.state)
    logger.info(f"Config: {cfg}")


    if accelerator.is_main_process:
        os.makedirs(proj_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=os.path.join(proj_dir, 'config.yaml'))
        accelerator.init_trackers(project_name=cfg.project_name, 
                                  config=OmegaConf.to_container(cfg, resolve=True), 
                                  init_kwargs={'wandb': {'name': cfg.exp_name, }})


    total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * \
                        cfg.training.gradient_accumulation_steps
    logger.info('***** Running training *****')
    logger.info(f'  Instantaneous batch size per device = {cfg.training.train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {cfg.training.max_iters}')


    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.training.num_train_timesteps,
                                                shift=cfg.training.fm_shift)

    from core.models.transformer_emmdit import EMMDiTTransformer
    model = EMMDiTTransformer(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            sample_size=cfg.model.latent_size,
            patch_size=cfg.model.patch_size,
            caption_channels=cfg.model.caption_channels,
            qk_norm='rms_norm',
            repa_depth=cfg.model.repa_depth,
            projector_dim=cfg.model.projector_dim,
            z_dims=cfg.model.z_dims,
            use_sub_attn = True,
        )

    if cfg.model.repa_depth != -1 and cfg.dataset.precompute_dino_feat is False:
        import timm
        visual_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
        del visual_encoder.head
        visual_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            visual_encoder.pos_embed.data, [16, 16],
        )
        visual_encoder.head = torch.nn.Identity()
        visual_encoder = visual_encoder.to(accelerator.device)
        visual_encoder.eval()
        visual_encoder = torch.compile(visual_encoder)
        
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_text_encoder = AutoModelForCausalLM.from_pretrained(cfg.llama_path, torch_dtype=torch.bfloat16)
    model_text_encoder.requires_grad_(False)
    model_text_encoder = model_text_encoder.to(accelerator.device)
    model_text_encoder = torch.compile(model_text_encoder)


    inputs = tokenizer('', return_tensors="pt", padding='max_length', max_length=cfg.model.caption_max_seq_length, truncation=True)
    inputs.to(accelerator.device)
    uncond_prompt_embeds = model_text_encoder(**inputs, output_hidden_states=True)['hidden_states'][-1]
    uncond_prompt_attention_mask = inputs['attention_mask']

    if cfg.model.flashSA is not None:
        if cfg.model.flashSA == 'Joint_SA':
            from core.models.flash_attn_processor import JointAttnProcessor2_0_FA
            model.set_attn_processor(JointAttnProcessor2_0_FA()) 

    if cfg.training.transformer_ckpt != '':
        state_dict = load_file(cfg.training.transformer_ckpt)
        if cfg.model.repa_depth == -1:
            new_state_dict = {}
            for k in state_dict:
                if 'projector' not in k:
                    new_state_dict[k] = state_dict[k]
            state_dict = new_state_dict
        model.load_state_dict(state_dict)

    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(model)


    if cfg.training.use_ema:
        model_ema = deepcopy(model).eval() 

    # mixed_precition training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # build dataloader
    dataloader_kwargs = {
        "batch_size": cfg.training.train_batch_size,
        "shuffle": True,
        "num_workers": cfg.training.num_workers,
        "pin_memory": True,
    }
    if cfg.dataset.use_dummy_data:
        from core.dataset import DummyDataset
        dataset = DummyDataset(
            latent_size=cfg.model.latent_size,
            caption_max_seq_length=cfg.model.caption_max_seq_length,
            caption_channels=cfg.model.caption_channels,
        )
        train_dataloader = DataLoader(dataset, **dataloader_kwargs)
    else:
        from core.dataset import build_streaming_latents_dataloader
        train_dataloader = build_streaming_latents_dataloader(
            cfg.dataset, 
            latent_size=cfg.model.latent_size,
            caption_max_seq_length=cfg.model.caption_max_seq_length,
            caption_channels=cfg.model.caption_channels,
            **dataloader_kwargs,
        )

    # Prepare everything
    if cfg.training.use_ema:
        model, model_ema = accelerator.prepare(model, model_ema) 
    else:
        model = accelerator.prepare(model)

    # build optimizer and lr scheduler
    opt_class, opt_kwargs = build_opt(cfg.optimizer.type, cfg.optimizer.opt_kwargs)
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    logger.info('optimizer info')
    logger.info(optimizer)

    # set lr scheduler
    lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.training.num_warmup_steps,
        )

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    global_step = 0

    latest_path = os.path.join(proj_dir, 'checkpoints', 'checkpoint-latest')
    if os.path.exists(latest_path):
        accelerator.load_state(latest_path)
        logger.info(f'resume training from {cfg.training.resume_from}')
        from glob import glob
        step_info = glob(os.path.join(latest_path, 'step-*.info'))[0]
        global_step = int(step_info.split('-')[-1][:-5]) + 1
        logger.info(f'Resuming from global step {global_step}')

    if cfg.training.resume_from != '':
        accelerator.load_state(cfg.training.resume_from)
        logger.info(f'resume training from {cfg.training.resume_from}')
        global_step = int(cfg.training.resume_from.split('-')[-1]) + 1
        logger.info(f'Resuming from global step {global_step}')

    progress_bar = tqdm(
        total=cfg.training.max_iters,
        initial=global_step,
        desc='Steps',
        disable=not accelerator.is_local_main_process,
    )


    # start training
    while True:
       
        for _, batch in enumerate(train_dataloader):
            """ #use dummydata
            image_latents, txt_emb, txt_mask, zs, jpg_tensors = batch
            txt_emb = txt_emb.to(accelerator.device)
            txt_mask = txt_mask.to(accelerator.device)
            zs = zs.to(accelerator.device)
            jpg_tensors = jpg_tensors.to(accelerator.device)
            """
            
            image_latents, prompts, jpg_tensors, txt_emb, txt_mask, dino_feat = batch
            dino_feat = dino_feat.to(accelerator.device)
            zs = [dino_feat]
            image_latents = image_latents.to(accelerator.device)
            
            if cfg.dataset.precompute_txt_emb is False:
                with torch.no_grad():
                    inputs = tokenizer(prompts, return_tensors="pt", padding='max_length', max_length=cfg.model.caption_max_seq_length, truncation=True)
                    inputs.to(accelerator.device)
                    txt_emb = model_text_encoder(**inputs, output_hidden_states=True)['hidden_states'][-1] #torch.Size([1, 128, 2048])
                    txt_mask = inputs['attention_mask'] #torch.Size([1, 128])             
                
            if cfg.dataset.precompute_dino_feat is False:
                with torch.no_grad():
                    if cfg.model.repa_depth != -1:
                        jpg_tensors = jpg_tensors.to(accelerator.device)
                        z = visual_encoder.forward_features(jpg_tensors.cuda())
                        z = z['x_norm_patchtokens'] #torch.Size([bs, 256, 768])
                
                        zs = [z]
                    

            latents = (image_latents * cfg.training.scaling_factor).to(weight_dtype)
            txt_mask = txt_mask.to(accelerator.device)
            
            y = txt_emb.to(weight_dtype).to(accelerator.device)
            y_mask = txt_mask.to(weight_dtype)
            y, y_mask = token_drop(y, y_mask,
                                   uncond_prompt_embeds,
                                   uncond_prompt_attention_mask,
                                   cfg.training.class_dropout_prob)

            bs = latents.shape[0]
            noise = torch.randn_like(latents)

            # timestep sampling and mix latent with noise
            u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.training.weighting_scheme,
                    batch_size=bs,
                    logit_mean=cfg.training.logit_mean,
                    logit_std=cfg.training.logit_std,
                    mode_scale=cfg.training.mode_scale,
                )
            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices]
            sigmas = get_sigmas(timesteps, noise_scheduler, n_dim=latents.ndim).to(device=accelerator.device)
            timesteps = timesteps.to(device=accelerator.device)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise


            grad_norm = None
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                model_pred, zs_tilde = model(hidden_states=noisy_latents,
                                    encoder_hidden_states=y,
                                    timestep=timesteps,
                                    encoder_attention_mask=y_mask,
                                    added_cond_kwargs={'resolution': None, 'aspect_ratio': None},
                                    return_dict=False)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=cfg.training.weighting_scheme, sigmas=sigmas)
                target = noise - latents

                
                model_pred = einops.rearrange(model_pred, 'n t (p1 p2 c) -> n c t (p1 p2)', 
                                                p1=cfg.model.patch_size, p2=cfg.model.patch_size)

                target = patchify_and_apply_mask(target, cfg.model.patch_size, )

                diff_loss  = calc_diff_loss(model_pred, target, weighting)


                if zs_tilde is not None:
                    proj_loss = calc_proj_loss(zs, zs_tilde, )
                    loss = diff_loss + proj_loss * cfg.training.proj_coeff
                else:
                    loss = diff_loss
                
                accelerator.backward(loss.contiguous())
                if accelerator.sync_gradients:
                    if accelerator.distributed_type == DistributedType.FSDP:
                        grad_norm = accelerator._models[0].clip_grad_norm_(cfg.training.gradient_clip, 2)
                    else:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                    if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()):
                        optimizer.zero_grad(set_to_none=True)
                        logger.warning("NaN or Inf detected in grad_norm, skipping iteration...")
                        continue

                optimizer.step()
                lr_scheduler.step()

                if cfg.training.use_ema:
                    if accelerator.sync_gradients: 
                        ema_update(accelerator.unwrap_model(model_ema), accelerator.unwrap_model(model), cfg.training.ema_rate)

            logs = {'loss': accelerator.gather(diff_loss).mean().item(), 
                        'lr': lr_scheduler.get_last_lr()[0]}
            if zs_tilde is not None:
                logs['proj_loss'] = accelerator.gather(proj_loss).mean().item()
            if grad_norm is not None:
                logs['grad_norm'] = accelerator.gather(grad_norm).mean().item()

            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            global_step += 1
            progress_bar.update(1)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if global_step % 2000 == 0:
                    latest_path = os.path.join(proj_dir, 'checkpoints', 'checkpoint-latest')
                    if os.path.exists(latest_path):
                        import glob
                        info_files = glob.glob(os.path.join(latest_path, 'step-*.info'))
                        for file_path in info_files:
                            os.remove(file_path)
                    logger.info(f"Start to save state to {latest_path}")
                    accelerator.save_state(latest_path)
                    with open(os.path.join(latest_path, 'step-%d.info'%global_step), "w") as f:
                        pass
                    logger.info(f"Saved state to {latest_path}")

                if global_step % cfg.training.save_freq == 0:
                    save_path = os.path.join(os.path.join(proj_dir, 'checkpoints'), f"checkpoint-{global_step}")
                    logger.info(f"Start to save state to {save_path}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if global_step and global_step % cfg.validation.validation_frequency == 0:
                    if accelerator.distributed_type == DistributedType.FSDP:
                        model_state_dict = model.state_dict()
                    else:
                        model_state_dict = accelerator.unwrap_model(model).state_dict()
                    
                    torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

            if global_step >= cfg.training.max_iters:
                break

        if global_step >= cfg.training.max_iters:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(os.path.join(proj_dir, 'checkpoints'), f"checkpoint-{global_step}")
        logger.info(f"Start to save state to {save_path}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
    
