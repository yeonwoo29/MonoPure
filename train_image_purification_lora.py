"""
LoRA finetuning for Stable Diffusion img2img purification using paired (recon -> clean) RGB images.

Assumptions:
- If --config and --recon-ckpt are provided, recon/clean pairs are generated on-the-fly:
    * recon: MonoPure backbone + reconstruction decoder output (normalized clamp[0,1])
    * clean: MonoPure input de-normalized to [0,1] (KITTI transformed image)
- Otherwise, recon_dir / clean_dir are used as pre-rendered folders with matching filenames.

Loss (LCM-style with anchor, inspired by train_lora):
    noisy(recon_latent, t) -> UNet -> pred_noise
    target_noise = noise (epsilon)
    anchor: pred_noise vs pred_noise_clean (noisy(clean_latent, t) through the same UNet, no grad)
    loss = mse(pred_noise, target_noise) + lambda_anchor * mse(pred_noise, target_noise_clean)

Example (on-the-fly KITTI):
  python tools/train_purify_lora.py \
    --config configs/monopure.yaml \
    --recon-ckpt outputs/backbone_reconstruction/reconstruction_decoder_best.pth \
    --output_dir outputs/purify_lora_sd15 \
    --pretrained_model runwayml/stable-diffusion-v1-5 \
    --train_batch_size 1 --num_train_epochs 3 --lambda-anchor 0.5
"""
import os
import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from contextlib import nullcontext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import yaml
from lib.helpers.dataloader_helper import build_dataloader
from lib.models.monopure.backbone import build_backbone
from lib.models.monopure.reconstruction_decoder import ReconstructionDecoder


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    img = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))).view(size, size, 3)
    img = img.permute(2, 0, 1).float() / 255.0
    return img


class ReconCleanPairDataset(Dataset):
    def __init__(self, recon_dir, clean_dir, train_size=512):
        self.recon_dir = Path(recon_dir)
        self.clean_dir = Path(clean_dir)
        self.train_size = train_size
        names = sorted([p.name for p in self.recon_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        self.pairs = []
        for n in names:
            if (self.clean_dir / n).exists():
                self.pairs.append((self.recon_dir / n, self.clean_dir / n))
        if len(self.pairs) == 0:
            raise RuntimeError("No matching recon/clean image pairs found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rpath, cpath = self.pairs[idx]
        recon = load_image(rpath, self.train_size)
        clean = load_image(cpath, self.train_size)
        return recon, clean


def parse_args():
    p = argparse.ArgumentParser("LoRA finetune SD img2img for purification")
    p.add_argument("--recon_dir", required=False, type=str, help="(optional) pre-rendered recon images")
    p.add_argument("--clean_dir", required=False, type=str, help="(optional) paired clean images")
    p.add_argument("--config", type=str, default=None, help="MonoPure config yaml for on-the-fly recon/clean")
    p.add_argument("--recon_ckpt", type=str, default=None, help="checkpoint containing backbone/decoder state_dict")
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--train_size", type=int, default=320)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--vae_encode_batch_size", type=int, default=1, help="chunk size for VAE.encode to save memory")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--adam_weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--vae_offload_cpu", action="store_true", help="move VAE to CPU to save GPU memory (slower)")
    p.add_argument("--lambda-anchor", type=float, default=0.5, help="weight for anchor loss against clean target")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--checkpointing_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--enable_gradient_checkpointing", action="store_true", help="enable UNet gradient checkpointing")
    p.add_argument("--enable_attention_slicing", action="store_true", help="enable UNet attention slicing")
    return p.parse_args()


def save_lora(unet, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
    torch.save(lora_state_dict, os.path.join(out_dir, "pytorch_lora_weights.bin"))
    print(f"[SAVE] LoRA weights -> {out_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")),
        split_batches=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # LoRA wrap
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "conv1", "conv2", "conv_shortcut",
            "downsamplers.0.conv", "upsamplers.0.conv",
            "time_emb_proj",
        ],
        lora_dropout=args.lora_dropout,
    )
    unet = get_peft_model(unet, lora_config)

    # optional memory savers
    if args.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.enable_attention_slicing:
        unet.enable_attention_slicing()

    # only LoRA params trainable
    for p in unet.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)
    trainable = [p for p in unet.parameters() if p.requires_grad]

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae_device = accelerator.device
    if args.vae_offload_cpu:
        vae_device = torch.device("cpu")
    vae_dtype = weight_dtype if vae_device.type == "cuda" else torch.float32
    vae.to(vae_device, dtype=vae_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=torch.float32)

    # data source: on-the-fly recon/clean via MonoPure backbone+decoder, else pre-rendered folders
    use_on_the_fly = args.config is not None and args.recon_ckpt is not None
    recon_backbone = decoder = None
    mean = std = None
    if use_on_the_fly:
        cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
        device = accelerator.device
        train_dataloader, _ = build_dataloader(cfg["dataset"])
        recon_backbone = build_backbone(cfg["model"]).to(device)
        decoder = ReconstructionDecoder().to(device)
        ckpt = torch.load(args.recon_ckpt, map_location=device)
        if "backbone" in ckpt:
            recon_backbone.load_state_dict(ckpt["backbone"], strict=False)
        if "decoder" in ckpt:
            decoder.load_state_dict(ckpt["decoder"], strict=False)
        recon_backbone.eval()
        decoder.eval()

        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    else:
        dataset = ReconCleanPairDataset(args.recon_dir, args.clean_dir, train_size=args.train_size)
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            drop_last=True,
        )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    # empty prompt embeddings
    with torch.no_grad():
        empty_ids = tokenizer([""], return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length).input_ids
        empty_ids = empty_ids.to(accelerator.device)
        text_embeds = text_encoder(empty_ids)[0]

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps")
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if use_on_the_fly:
                    inputs, _, _, _ = batch
                    # keep backbone in fp32 to avoid dtype mismatch
                    inputs = inputs.to(accelerator.device, dtype=torch.float32)
                    with torch.no_grad():
                        feats, _ = recon_backbone(inputs)
                        f2, f3, f4 = feats[0].tensors, feats[1].tensors, feats[2].tensors
                        recon = decoder(f2, f3, f4)
                        if recon.shape[-2:] != inputs.shape[-2:]:
                            recon = torch.nn.functional.interpolate(
                                recon, size=inputs.shape[-2:], mode="bilinear", align_corners=False
                            )
                        recon = recon.clamp(0, 1)
                        clean = (inputs * std + mean).clamp(0, 1)
                    # convert to weight_dtype for VAE/UNet
                    recon = recon.to(accelerator.device, dtype=weight_dtype)
                    clean = clean.to(accelerator.device, dtype=weight_dtype)
                else:
                    recon, clean = batch
                    recon = recon.to(accelerator.device, dtype=weight_dtype)
                    clean = clean.to(accelerator.device, dtype=weight_dtype)

                # [-1,1]
                recon_in = recon * 2.0 - 1.0
                clean_in = clean * 2.0 - 1.0

                # encode to latents with chunked VAE to save memory
                def encode_in_chunks(x):
                    outs = []
                    for i in range(0, x.shape[0], args.vae_encode_batch_size):
                        chunk = x[i: i + args.vae_encode_batch_size]
                        if vae_device.type == "cpu":
                            chunk = chunk.to(vae_device, dtype=torch.float32)
                            ctx = nullcontext
                        else:
                            chunk = chunk.to(vae_device, dtype=vae_dtype)
                            ctx = torch.cuda.amp.autocast if vae_dtype != torch.float32 else nullcontext
                        with ctx():
                            outs.append(vae.encode(chunk).latent_dist.sample())
                    lat = torch.cat(outs, dim=0) * vae.config.scaling_factor
                    return lat.to(accelerator.device, dtype=weight_dtype)

                latents_src = encode_in_chunks(recon_in)
                latents_tgt = encode_in_chunks(clean_in)

                bsz = latents_src.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_src.device).long()

                noise = torch.randn_like(latents_src)
                noisy_latents = noise_scheduler.add_noise(latents_src, noise, timesteps)

                # target noise derived from target latents (clean)
                noisy_tgt = noise_scheduler.add_noise(latents_tgt, noise, timesteps)
                # main target: epsilon (noise) for epsilon prediction_type
                target_noise = noise
                # anchor target: predicted noise from clean latent (no grad)
                with torch.no_grad():
                    target_pred = unet(noisy_tgt, timesteps, encoder_hidden_states=text_embeds).sample

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample
                loss_main = F.mse_loss(model_pred.float(), target_noise.float())
                loss_anchor = F.mse_loss(model_pred.float(), target_pred.float())
                loss = loss_main + args.lambda_anchor * loss_anchor

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process and global_step % args.logging_steps == 0:
                    msg = f"epoch {epoch+1} step {global_step} loss {loss.item():.6f}"
                    tqdm.write(msg)
                    accelerator.log({"loss": loss.item()}, step=global_step)
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_lora(accelerator.unwrap_model(unet), os.path.join(args.output_dir, f"unet_lora_step{global_step}"))

            if global_step >= max_train_steps:
                break
            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_ = accelerator.unwrap_model(unet)
        save_lora(unet_, os.path.join(args.output_dir, "unet_lora_final"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
