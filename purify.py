# purify_only.py
# Pure purification for a folder of (possibly adversarial) images.
# Keeps original resolution (no resize), uses ControlNet-Canny + SD v1.5 with LCM/TCD.
# Usage:
#   python purify_only.py --model TCD --input_dir /mnt/d/data/kitti/kitti_attack/kitti_val_e32_pgd/ --output_dir '/mnt/d/data/kitti/kitti_puri/(baseline_pgd)' --lora_input_dir /mnt/d/실험/purify/InstantPure/logs/OSCP/unet_lora/ --num_inference_step 5 --strength 0.2 --guidance_scale 1.0 --control_scale 0 --device cuda:0

import argparse, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler,
)
try:
    from diffusers import TCDScheduler
except ImportError:
    TCDScheduler = None

LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "proj_in",
    "proj_out",
    "ff.net.0.proj",
    "ff.net.2",
    "conv1",
    "conv2",
    "conv_shortcut",
    "downsamplers.0.conv",
    "upsamplers.0.conv",
    "time_emb_proj",
)

def parse_args():
    p = argparse.ArgumentParser("Purify a folder of images with ControlNet-Canny + SD v1.5")
    p.add_argument("--model", required=True, type=str, help="LCM or TCD")
    p.add_argument("--lora_input_dir", type=str, default="latent-consistency/lcm-lora-sdv1-5",
                   help="LoRA repo or local path, set empty to skip")
    p.add_argument("--input_dir", required=True, type=str, help="input image dir (clean or adversarial)")
    p.add_argument("--output_dir", required=True, type=str, help="where to save purified images")
    p.add_argument("--num_inference_step", type=int, default=5, help="diffusion steps")
    p.add_argument("--strength", type=float, default=0.2, help="img2img noise strength")
    p.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale")
    p.add_argument("--control_scale", type=float, default=0.8, help="ControlNet conditioning scale")
    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--canny_low", type=int, default=100)
    p.add_argument("--canny_high", type=int, default=200)
    return p.parse_args()

def _count_params(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())

def _count_state_dict_params(state_dict, include_prefixes=None, exclude_prefixes=None):
    total = 0
    for k, v in state_dict.items():
        if include_prefixes and not any(k.startswith(p) for p in include_prefixes):
            continue
        if exclude_prefixes and any(k.startswith(p) for p in exclude_prefixes):
            continue
        if hasattr(v, "numel"):
            total += v.numel()
    return total

def _match_lora_target(key, target_name):
    needle = f".{target_name}."
    return needle in key or key.endswith(f".{target_name}")

def _summarize_lora_modules(state_dict, target_modules, include_prefixes=None):
    total = 0
    per_module = {name: 0 for name in target_modules}
    other = 0
    for k, v in state_dict.items():
        if include_prefixes and not any(k.startswith(p) for p in include_prefixes):
            continue
        if not hasattr(v, "numel"):
            continue
        total += v.numel()
        matched = False
        for name in target_modules:
            if _match_lora_target(k, name):
                per_module[name] += v.numel()
                matched = True
                break
        if not matched:
            other += v.numel()
    return total, per_module, other

def pad_to_multiple(h: int, w: int, m: int = 8):
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    pl, pr = pw // 2, pw - pw // 2
    pt, pb = ph // 2, ph - ph // 2
    return pl, pr, pt, pb

def unsharp_mask_pt(x: torch.Tensor, amount: float = 0.15, radius: int = 3, threshold: float = 0.02):
    """
    x: [B,3,H,W] in [0,1]
    amount: 샤프 강도 (약하게: 0.1~0.2 권장)
    radius: 블러 커널 크기(홀수) (3 or 5 권장)
    threshold: 너무 미세한 변화(노이즈)엔 샤프 적용 안 함
    """
    if amount <= 0:
        return x

    # 안전장치
    radius = int(radius)
    if radius < 1:
        return x
    if radius % 2 == 0:
        radius += 1

    # 간단한 box blur (빠르고 안정적)
    blurred = F.avg_pool2d(x, kernel_size=radius, stride=1, padding=radius // 2)

    # high-frequency 성분
    high = x - blurred

    # threshold 아래는 0으로 (노이즈 과증폭 방지)
    if threshold > 0:
        mask = (high.abs().mean(dim=1, keepdim=True) > threshold).to(x.dtype)
        high = high * mask

    y = x + amount * high
    return y.clamp(0, 1)



@torch.inference_mode()
def purify_tensor(img_pt: torch.Tensor,
                  pipe: StableDiffusionControlNetImg2ImgPipeline,
                  canny_low: int, canny_high: int,
                  num_steps: int, strength: float,
                  guidance_scale: float, control_scale: float,
                  seed: int,
                  return_timing: bool = False):
    """
    img_pt: [1,3,H,W] in [0,1], device==pipe device
    returns: purified [1,3,H,W] in [0,1]
    """
    _, _, H, W = img_pt.shape

    # Build Canny control (H,W,3) without resizing
    pil = TF.to_pil_image(img_pt.squeeze(0).detach().cpu())
    np_img = np.array(pil)
    edges = cv2.Canny(np_img, canny_low, canny_high)
    edges3 = np.repeat(edges[..., None], 3, axis=2)
    control_pil = Image.fromarray(edges3).convert("RGB")

    # Pad to multiple of 8 (diffusers UNet/ControlNet friendly)
    pl, pr, pt, pb = pad_to_multiple(H, W, m=8)
    if any([pl, pr, pt, pb]):
        img_pad = F.pad(img_pt, (pl, pr, pt, pb), mode="reflect")  # [1,3,H',W']
        ctrl_pad = F.pad(TF.to_tensor(control_pil), (pl, pr, pt, pb), mode="reflect")
        control_pil = TF.to_pil_image(ctrl_pad).convert("RGB")
    else:
        img_pad = img_pt

    # ------------------------------------------------------------
    # [PATCH] avoid empty timesteps by ensuring >= 1 denoising step
    # but keep "effective change" small via blending.
    # ------------------------------------------------------------
    # diffusers 내부에서 int(num_steps * strength) == 0 이면 터질 수 있음
    # -> 최소 1 step이 되도록 strength_eff를 올림
    min_strength = 1.0 / max(int(num_steps), 1)
    strength_eff = max(float(strength), float(min_strength) + 1e-8)

    # 실제 정화량은 원래 strength에 맞춰 약하게 유지하기 위한 블렌딩 계수
    # (strength < strength_eff 일 때 alpha < 1)
    alpha = float(strength) / float(strength_eff)
    alpha = max(0.0, min(1.0, alpha))

    # Generator (device 지정해주는 게 안전)
    try:
        generator = torch.Generator(device=img_pad.device).manual_seed(int(seed))
    except TypeError:
        generator = torch.manual_seed(int(seed))

    # Run img2img purification pass
    if img_pad.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(
        prompt=[""],
        image=img_pad,
        control_image=control_pil,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength_eff),  # <-- use effective strength to avoid crash
        controlnet_conditioning_scale=float(control_scale),
        generator=generator,
        output_type="pt",
        return_dict=False
    )
    if img_pad.is_cuda:
        torch.cuda.synchronize()
    diffusion_time = time.perf_counter() - t0
    out_pt = out[0]  # [1,3,H',W'] in [0,1]
    
    
    
    
    # --- NEW: very mild sharpening (USM) on diffusion output ---
    # 약하게: amount 0.10~0.20, radius 3, threshold 0.02 정도 추천
    #out_pt = unsharp_mask_pt(out_pt, amount=0.1, radius=3, threshold=0.02)




    # [PATCH] Blend back toward original to keep changes small
    # out_pt and img_pad should be same dtype/device; cast just in case.
    if out_pt.dtype != img_pad.dtype:
        out_pt = out_pt.to(img_pad.dtype)
    out_pt = alpha * out_pt + (1.0 - alpha) * img_pad
    out_pt = out_pt.clamp(0, 1)

    # Unpad back to original size
    if any([pl, pr, pt, pb]):
        out_pt = out_pt[..., pt:pt+H, pl:pl+W]
    if return_timing:
        return out_pt, diffusion_time
    return out_pt


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" in args.device else "cpu")

    # Build pipeline once
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        variant="fp16",
    ).to(device)
    try:
        pipe.disable_xformers_memory_efficient_attention()
        print("xformers memory efficient attention disabled")
    except Exception:
        pass

    # Scheduler
    if args.model.upper() == "LCM":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif args.model.upper() == "TCD":
        if TCDScheduler is None:
            raise ImportError(
                "TCDScheduler not available in your diffusers version. "
                "Upgrade diffusers (pip install -U diffusers) or use --model LCM."
            )
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError("model must be LCM or TCD")

    # Optional LoRA
    trainable_params_m = 0.0
    trainable_module_params_m = {name: 0.0 for name in LORA_TARGET_MODULES}
    trainable_other_params_m = 0.0
    if args.lora_input_dir:
        try:
            # Mirror test.py: unwrap PEFT keys so adapter_config.json is not required alongside the weights.
            lora_state_dict = pipe.lora_state_dict(args.lora_input_dir)
            unwrapped_state_dict = {}

            for peft_key, weight in lora_state_dict[0].items():
                key = peft_key.replace("base_model.model.", "")
                unwrapped_state_dict[key] = weight.to(pipe.dtype)

            pipe.load_lora_weights(unwrapped_state_dict, adapter_name="default")
            lora_total, lora_per_module, lora_other = _summarize_lora_modules(
                unwrapped_state_dict,
                LORA_TARGET_MODULES,
                include_prefixes=("unet", "controlnet"),
            )
            trainable_params_m = lora_total / 1e6
            trainable_module_params_m = {k: v / 1e6 for k, v in lora_per_module.items()}
            trainable_other_params_m = lora_other / 1e6
            print(f"Loaded LoRA from {args.lora_input_dir}")
        except Exception as e:
            print(f"[WARN] Failed to load LoRA from {args.lora_input_dir}: {e}")

    params_m = sum(_count_params(m) for m in [
        pipe.unet, pipe.controlnet
    ]) / 1e6
    params_full_m = params_m + sum(_count_params(m) for m in [
        pipe.vae, getattr(pipe, "text_encoder", None)
    ]) / 1e6

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png",".jpg",".jpeg",".bmp",".webp"}
    paths = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts])
    if not paths:
        raise FileNotFoundError(f"No images found under {in_dir}")

    flops_total_g = None
    try:
        import torch.profiler as prof
        sample = Image.open(paths[0]).convert("RGB")
        x = TF.to_tensor(sample).unsqueeze(0).to(device)
        activities = [prof.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(prof.ProfilerActivity.CUDA)
        with prof.profile(activities=activities, with_flops=True, record_shapes=True) as p:
            _ = purify_tensor(
                x, pipe,
                canny_low=args.canny_low, canny_high=args.canny_high,
                num_steps=args.num_inference_step,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                control_scale=args.control_scale,
                seed=args.seed
            )
        flops = sum(e.flops for e in p.key_averages() if getattr(e, "flops", None))
        flops_total_g = flops / 1e9
    except Exception as e:
        print(f"[WARN] FLOPs profiling failed: {e}")

    flops_g = None
    if flops_total_g is not None:
        if params_full_m > 0:
            flops_g = flops_total_g * (params_m / params_full_m)
        else:
            flops_g = flops_total_g

    trainable_flops_g = None
    if flops_g is not None and params_m > 0:
        trainable_flops_g = flops_g * (trainable_params_m / params_m)

    trainable_module_flops_g = {name: None for name in LORA_TARGET_MODULES}
    if trainable_flops_g is not None and trainable_params_m > 0:
        for name in LORA_TARGET_MODULES:
            module_params_m = trainable_module_params_m.get(name, 0.0)
            trainable_module_flops_g[name] = trainable_flops_g * (module_params_m / trainable_params_m)

    flops_str = f"{flops_g:.2f}" if flops_g is not None else "N/A"
    trainable_flops_str = f"{trainable_flops_g:.4f}" if trainable_flops_g is not None else "N/A"
    print(
        f"[Approx] Params (M): {params_m:.2f} | FLOPs (G): {flops_str} | "
        f"Trainable Params (M): {trainable_params_m:.2f} | Trainable FLOPs (G): {trainable_flops_str}",
        flush=True,
    )
    if trainable_params_m > 0:
        # print("[Approx] Trainable breakdown by LoRA target module:", flush=True)
        # for name in sorted(LORA_TARGET_MODULES, key=lambda n: trainable_module_params_m.get(n, 0.0), reverse=True):
        #     module_params_m = trainable_module_params_m.get(name, 0.0)
        #     if module_params_m == 0:
        #         continue
        #     module_flops_g = trainable_module_flops_g.get(name)
        #     module_flops_str = f"{module_flops_g:.4f}" if module_flops_g is not None else "N/A"
        #     print(f"  {name}: Params (M): {module_params_m:.4f} | FLOPs (G): {module_flops_str}", flush=True)
        # if trainable_other_params_m > 0:
        #     other_flops_g = None
        #     if trainable_flops_g is not None:
        #         other_flops_g = trainable_flops_g * (trainable_other_params_m / trainable_params_m)
        #     other_flops_str = f"{other_flops_g:.4f}" if other_flops_g is not None else "N/A"
        #     print(
        #         f"  other: Params (M): {trainable_other_params_m:.4f} | FLOPs (G): {other_flops_str}",
        #         flush=True,
        #     )

    t0 = time.time()
    total_time = 0.0
    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
            x = TF.to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W], [0,1]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()

            y, diffusion_time = purify_tensor(
                x, pipe,
                canny_low=args.canny_low, canny_high=args.canny_high,
                num_steps=args.num_inference_step,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                control_scale=args.control_scale,
                seed=args.seed,
                return_timing=True
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            total_time += elapsed
            print(f"diffusion time {diffusion_time:.5f}")
            print(f"done in {elapsed:.5f}")
            if i == 1:
                latency_ms = total_time / i * 1000
                flops_str = f"{flops_g:.2f}" if flops_g is not None else "N/A"
                trainable_flops_str = f"{trainable_flops_g:.4f}" if trainable_flops_g is not None else "N/A"
                print(
                    f"Params (M): {params_m:.2f} | FLOPs (G): {flops_str} | "
                    f"Trainable Params (M): {trainable_params_m:.2f} | Trainable FLOPs (G): {trainable_flops_str} | "
                    f"Latency (ms): {latency_ms:.2f}"
                )

            # Save with same relative structure under output_dir
            rel = p.relative_to(in_dir)
            save_path = (out_dir / rel).with_suffix(".png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            TF.to_pil_image(y.squeeze(0).clamp(0,1).cpu()).save(save_path)
            print(f"[{i}/{len(paths)}] saved -> {save_path}")
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    print(f"done in {time.time()-t0:.5f}s  images={len(paths)}  out={out_dir}")
    latency_ms = (total_time / len(paths) * 1000) if paths else 0.0
    flops_str = f"{flops_g:.2f}" if flops_g is not None else "N/A"
    trainable_flops_str = f"{trainable_flops_g:.4f}" if trainable_flops_g is not None else "N/A"
    print(
        f"Params (M): {params_m:.2f} | FLOPs (G): {flops_str} | "
        f"Trainable Params (M): {trainable_params_m:.2f} | Trainable FLOPs (G): {trainable_flops_str} | "
        f"Latency (ms): {latency_ms:.2f}"
    )

if __name__ == "__main__":
    main()















# import argparse, os, time
# from pathlib import Path
# import torch
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
# import kornia as K
# from PIL import Image
# from diffusers import (
#     StableDiffusionControlNetImg2ImgPipeline,
#     ControlNetModel,
#     LCMScheduler,
# )
# try:
#     from diffusers import TCDScheduler
# except ImportError:
#     TCDScheduler = None

# LORA_TARGET_MODULES = (
#     "to_q",
#     "to_k",
#     "to_v",
#     "to_out.0",
#     # "proj_in",
#     # "proj_out",
#     # "ff.net.0.proj",
#     # "ff.net.2",
#     "conv1",
#     "conv2",
#     # "conv_shortcut",
#     # "downsamplers.0.conv",
#     # "upsamplers.0.conv",
#     # "time_emb_proj",
# )

# def parse_args():
#     p = argparse.ArgumentParser("Purify a folder of images with ControlNet-Canny + SD v1.5")
#     p.add_argument("--model", required=True, type=str, help="LCM or TCD")
#     p.add_argument("--lora_input_dir", type=str, default="latent-consistency/lcm-lora-sdv1-5",
#                    help="LoRA repo or local path, set empty to skip")
#     p.add_argument("--input_dir", required=True, type=str, help="input image dir (clean or adversarial)")
#     p.add_argument("--output_dir", required=True, type=str, help="where to save purified images")
#     p.add_argument("--num_inference_step", type=int, default=5, help="diffusion steps")
#     p.add_argument("--strength", type=float, default=0.2, help="img2img noise strength")
#     p.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale")
#     p.add_argument("--control_scale", type=float, default=0.8, help="ControlNet conditioning scale")
#     p.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
#     p.add_argument("--seed", type=int, default=3407)
#     p.add_argument("--canny_low", type=int, default=100)
#     p.add_argument("--canny_high", type=int, default=200)
#     p.add_argument("--batch_size", type=int, default=1, help="batch size for same-resolution images")
#     p.add_argument("--resize_scale", type=float, default=0.5,
#                    help="resize input resolution by this scale (1.0 = original)")
#     return p.parse_args()

# def _count_params(module):
#     if module is None:
#         return 0
#     return sum(p.numel() for p in module.parameters())

# def _count_state_dict_params(state_dict, include_prefixes=None, exclude_prefixes=None):
#     total = 0
#     for k, v in state_dict.items():
#         if include_prefixes and not any(k.startswith(p) for p in include_prefixes):
#             continue
#         if exclude_prefixes and any(k.startswith(p) for p in exclude_prefixes):
#             continue
#         if hasattr(v, "numel"):
#             total += v.numel()
#     return total

# def _match_lora_target(key, target_name):
#     needle = f".{target_name}."
#     return needle in key or key.endswith(f".{target_name}")

# def _scaled_size(w: int, h: int, scale: float):
#     w_out = max(1, int(round(w * scale)))
#     h_out = max(1, int(round(h * scale)))
#     return w_out, h_out

# def _summarize_lora_modules(state_dict, target_modules, include_prefixes=None):
#     total = 0
#     per_module = {name: 0 for name in target_modules}
#     other = 0
#     for k, v in state_dict.items():
#         if include_prefixes and not any(k.startswith(p) for p in include_prefixes):
#             continue
#         if not hasattr(v, "numel"):
#             continue
#         total += v.numel()
#         matched = False
#         for name in target_modules:
#             if _match_lora_target(k, name):
#                 per_module[name] += v.numel()
#                 matched = True
#                 break
#         if not matched:
#             other += v.numel()
#     return total, per_module, other

# def _find_lora_weights(module):
#     for down_name, up_name in (("lora_down", "lora_up"), ("lora_A", "lora_B"), ("down", "up")):
#         down = getattr(module, down_name, None)
#         up = getattr(module, up_name, None)
#         if hasattr(down, "weight") and hasattr(up, "weight"):
#             return down, up
#     lora_layer = getattr(module, "lora_layer", None)
#     if lora_layer is not None:
#         for down_name, up_name in (("lora_down", "lora_up"), ("down", "up"), ("lora_A", "lora_B")):
#             down = getattr(lora_layer, down_name, None)
#             up = getattr(lora_layer, up_name, None)
#             if hasattr(down, "weight") and hasattr(up, "weight"):
#                 return down, up
#     return None

# def _estimate_lora_flops_from_weights(x, down_w, up_w):
#     if not torch.is_tensor(x):
#         return 0.0
#     if down_w.dim() == 2 and up_w.dim() == 2:
#         in_features = down_w.shape[1]
#         r = down_w.shape[0]
#         out_features = up_w.shape[0]
#         tokens = x.numel() / max(in_features, 1)
#         flops = 2.0 * tokens * in_features * r
#         flops += 2.0 * tokens * r * out_features
#         return flops
#     if down_w.dim() == 4 and up_w.dim() == 4 and x.dim() == 4:
#         b, _, h, w = x.shape
#         in_ch = down_w.shape[1]
#         r = down_w.shape[0]
#         k1 = down_w.shape[2]
#         out_ch = up_w.shape[0]
#         k2 = up_w.shape[2]
#         flops = 2.0 * b * h * w * r * in_ch * k1 * k1
#         flops += 2.0 * b * h * w * out_ch * r * k2 * k2
#         return flops
#     return 0.0

# def _estimate_lora_flops(pipe, sample_pt, args):
#     total_flops = 0.0
#     hooks = []

#     def _register(model):
#         for name, module in model.named_modules():
#             if not any(_match_lora_target(name, t) for t in LORA_TARGET_MODULES):
#                 continue
#             pair = _find_lora_weights(module)
#             if pair is None:
#                 continue
#             down, up = pair

#             def _hook(_module, inputs, _output, down=down, up=up):
#                 nonlocal total_flops
#                 x = inputs[0] if inputs else None
#                 total_flops += _estimate_lora_flops_from_weights(x, down.weight, up.weight)

#             hooks.append(module.register_forward_hook(_hook))

#     _register(pipe.unet)
#     _register(pipe.controlnet)
#     try:
#         _ = purify_tensor(
#             sample_pt, pipe,
#             canny_low=args.canny_low, canny_high=args.canny_high,
#             num_steps=args.num_inference_step,
#             strength=args.strength,
#             guidance_scale=args.guidance_scale,
#             control_scale=args.control_scale,
#             seed=args.seed
#         )
#     finally:
#         for h in hooks:
#             h.remove()
#     return total_flops / 1e9

# def pad_to_multiple(h: int, w: int, m: int = 8):
#     ph = (m - h % m) % m
#     pw = (m - w % m) % m
#     pl, pr = pw // 2, pw - pw // 2
#     pt, pb = ph // 2, ph - ph // 2
#     return pl, pr, pt, pb

# def unsharp_mask_pt(x: torch.Tensor, amount: float = 0.15, radius: int = 3, threshold: float = 0.02):
#     """
#     x: [B,3,H,W] in [0,1]
#     amount: 샤프 강도 (약하게: 0.1~0.2 권장)
#     radius: 블러 커널 크기(홀수) (3 or 5 권장)
#     threshold: 너무 미세한 변화(노이즈)엔 샤프 적용 안 함
#     """
#     if amount <= 0:
#         return x

#     # 안전장치
#     radius = int(radius)
#     if radius < 1:
#         return x
#     if radius % 2 == 0:
#         radius += 1

#     # 간단한 box blur (빠르고 안정적)
#     blurred = F.avg_pool2d(x, kernel_size=radius, stride=1, padding=radius // 2)

#     # high-frequency 성분
#     high = x - blurred

#     # threshold 아래는 0으로 (노이즈 과증폭 방지)
#     if threshold > 0:
#         mask = (high.abs().mean(dim=1, keepdim=True) > threshold).to(x.dtype)
#         high = high * mask

#     y = x + amount * high
#     return y.clamp(0, 1)



# @torch.inference_mode()
# def purify_tensor(img_pt: torch.Tensor,
#                   pipe: StableDiffusionControlNetImg2ImgPipeline,
#                   canny_low: int, canny_high: int,
#                   num_steps: int, strength: float,
#                   guidance_scale: float, control_scale: float,
#                   seed: int):
#     """
#     img_pt: [B,3,H,W] in [0,1], device==pipe device
#     returns: purified [B,3,H,W] in [0,1]
#     """
#     bsz, _, H, W = img_pt.shape

#     # Build Canny control on GPU (no resize)
#     gray = K.color.rgb_to_grayscale(img_pt)
#     canny = K.filters.Canny(low_threshold=canny_low / 255.0, high_threshold=canny_high / 255.0)
#     edges, _ = canny(gray)
#     control_pt = edges.repeat(1, 3, 1, 1).clamp(0, 1)

#     # Pad to multiple of 8 (diffusers UNet/ControlNet friendly)
#     pl, pr, pt, pb = pad_to_multiple(H, W, m=8)
#     if any([pl, pr, pt, pb]):
#         img_pad = F.pad(img_pt, (pl, pr, pt, pb), mode="reflect")  # [1,3,H',W']
#         control_pt = F.pad(control_pt, (pl, pr, pt, pb), mode="reflect")
#     else:
#         img_pad = img_pt

#     # ------------------------------------------------------------
#     # [PATCH] avoid empty timesteps by ensuring >= 1 denoising step
#     # but keep "effective change" small via blending.
#     # ------------------------------------------------------------
#     # diffusers 내부에서 int(num_steps * strength) == 0 이면 터질 수 있음
#     # -> 최소 1 step이 되도록 strength_eff를 올림
#     min_strength = 1.0 / max(int(num_steps), 1)
#     strength_eff = max(float(strength), float(min_strength) + 1e-8)

#     # 실제 정화량은 원래 strength에 맞춰 약하게 유지하기 위한 블렌딩 계수
#     # (strength < strength_eff 일 때 alpha < 1)
#     alpha = float(strength) / float(strength_eff)
#     alpha = max(0.0, min(1.0, alpha))

#     # Generator (device 지정해주는 게 안전)
#     generators = []
#     for i in range(bsz):
#         try:
#             g = torch.Generator(device=img_pad.device).manual_seed(int(seed) + i)
#         except TypeError:
#             g = torch.manual_seed(int(seed) + i)
#         generators.append(g)

#     # Run img2img purification pass
#     out = pipe(
#         prompt=[""] * bsz,
#         image=img_pad,
#         control_image=control_pt,
#         num_inference_steps=int(num_steps),
#         guidance_scale=float(guidance_scale),
#         strength=float(strength_eff),  # <-- use effective strength to avoid crash
#         controlnet_conditioning_scale=float(control_scale),
#         generator=generators,
#         output_type="pt",
#         return_dict=False
#     )
#     out_pt = out[0]  # [1,3,H',W'] in [0,1]
    
    
    
    
#     # --- NEW: very mild sharpening (USM) on diffusion output ---
#     # 약하게: amount 0.10~0.20, radius 3, threshold 0.02 정도 추천
#     #out_pt = unsharp_mask_pt(out_pt, amount=0.1, radius=3, threshold=0.02)




#     # [PATCH] Blend back toward original to keep changes small
#     # out_pt and img_pad should be same dtype/device; cast just in case.
#     if out_pt.dtype != img_pad.dtype:
#         out_pt = out_pt.to(img_pad.dtype)
#     out_pt = alpha * out_pt + (1.0 - alpha) * img_pad
#     out_pt = out_pt.clamp(0, 1)

#     # Unpad back to original size
#     if any([pl, pr, pt, pb]):
#         out_pt = out_pt[..., pt:pt+H, pl:pl+W]
#     return out_pt


# def main():
#     args = parse_args()
#     torch.set_grad_enabled(False)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.benchmark = True

#     device = torch.device(args.device if torch.cuda.is_available() or "cuda" in args.device else "cpu")
#     resize_scale = float(args.resize_scale)
#     if resize_scale <= 0:
#         raise ValueError("resize_scale must be > 0")
#     print(f"Resize scale: {resize_scale:.3f}")

#     # Build pipeline once
#     controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
#     pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         controlnet=controlnet,
#         torch_dtype=torch.float16,
#         safety_checker=None,
#         variant="fp16",
#     ).to(device)
#     try:
#         pipe.enable_xformers_memory_efficient_attention()
#         print("xformers memory efficient attention enabled")
#     except Exception as e:
#         print(f"[WARN] xformers enable failed: {e}")

#     # Scheduler
#     if args.model.upper() == "LCM":
#         pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
#     elif args.model.upper() == "TCD":
#         if TCDScheduler is None:
#             raise ImportError(
#                 "TCDScheduler not available in your diffusers version. "
#                 "Upgrade diffusers (pip install -U diffusers) or use --model LCM."
#             )
#         pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
#     else:
#         raise ValueError("model must be LCM or TCD")

#     # Optional LoRA
#     trainable_params_m = 0.0
#     trainable_module_params_m = {name: 0.0 for name in LORA_TARGET_MODULES}
#     trainable_other_params_m = 0.0
#     if args.lora_input_dir:
#         try:
#             # Mirror test.py: unwrap PEFT keys so adapter_config.json is not required alongside the weights.
#             lora_state_dict = pipe.lora_state_dict(args.lora_input_dir)
#             unwrapped_state_dict = {}

#             for peft_key, weight in lora_state_dict[0].items():
#                 key = peft_key.replace("base_model.model.", "")
#                 unwrapped_state_dict[key] = weight.to(pipe.dtype)

#             pipe.load_lora_weights(unwrapped_state_dict, adapter_name="default")
#             lora_total, lora_per_module, lora_other = _summarize_lora_modules(
#                 unwrapped_state_dict,
#                 LORA_TARGET_MODULES,
#                 include_prefixes=("unet", "controlnet"),
#             )
#             trainable_module_params_m = {k: v / 1e6 for k, v in lora_per_module.items()}
#             trainable_params_m = sum(trainable_module_params_m.values())
#             trainable_other_params_m = lora_other / 1e6
#             print(f"Loaded LoRA from {args.lora_input_dir}")
#         except Exception as e:
#             print(f"[WARN] Failed to load LoRA from {args.lora_input_dir}: {e}")

#     params_m = sum(_count_params(m) for m in [
#         pipe.unet, pipe.controlnet
#     ]) / 1e6
#     params_full_m = params_m + sum(_count_params(m) for m in [
#         pipe.vae, getattr(pipe, "text_encoder", None)
#     ]) / 1e6

#     in_dir = Path(args.input_dir)
#     out_dir = Path(args.output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     exts = {".png",".jpg",".jpeg",".bmp",".webp"}
#     paths = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts])
#     if not paths:
#         raise FileNotFoundError(f"No images found under {in_dir}")

#     # Group by resolution to enable batching with same sizes
#     groups = {}
#     for p in paths:
#         try:
#             with Image.open(p) as im:
#                 w, h = im.size
#             rw, rh = _scaled_size(w, h, resize_scale)
#             groups.setdefault((rw, rh), []).append(p)
#         except Exception as e:
#             print(f"[WARN] Failed to read image size: {p}: {e}")

#     flops_total_g = None
#     try:
#         import torch.profiler as prof
#         sample = Image.open(paths[0]).convert("RGB")
#         if resize_scale != 1.0:
#             sw, sh = sample.size
#             rw, rh = _scaled_size(sw, sh, resize_scale)
#             sample = sample.resize((rw, rh), resample=Image.BILINEAR)
#         x = TF.to_tensor(sample).unsqueeze(0).to(device)
#         activities = [prof.ProfilerActivity.CPU]
#         if device.type == "cuda":
#             activities.append(prof.ProfilerActivity.CUDA)
#         with prof.profile(activities=activities, with_flops=True, record_shapes=True) as p:
#             _ = purify_tensor(
#                 x, pipe,
#                 canny_low=args.canny_low, canny_high=args.canny_high,
#                 num_steps=args.num_inference_step,
#                 strength=args.strength,
#                 guidance_scale=args.guidance_scale,
#                 control_scale=args.control_scale,
#                 seed=args.seed
#             )
#         flops = sum(e.flops for e in p.key_averages() if getattr(e, "flops", None))
#         flops_total_g = flops / 1e9
#     except Exception as e:
#         print(f"[WARN] FLOPs profiling failed: {e}")

#     flops_g = None
#     if flops_total_g is not None:
#         if params_full_m > 0:
#             flops_g = flops_total_g * (params_m / params_full_m)
#         else:
#             flops_g = flops_total_g

#     trainable_flops_g = None
#     if flops_g is not None and params_m > 0:
#         trainable_flops_g = flops_g * (trainable_params_m / params_m)

#     trainable_module_flops_g = {name: None for name in LORA_TARGET_MODULES}
#     if trainable_flops_g is not None and trainable_params_m > 0:
#         for name in LORA_TARGET_MODULES:
#             module_params_m = trainable_module_params_m.get(name, 0.0)
#             trainable_module_flops_g[name] = trainable_flops_g * (module_params_m / trainable_params_m)

#     flops_str = f"{flops_g:.2f}" if flops_g is not None else "N/A"
#     trainable_flops_str = f"{trainable_flops_g:.4f}" if trainable_flops_g is not None else "N/A"
#     print(
#         f"[Approx] Params (M): {params_m:.2f} | FLOPs (G): {flops_str} | "
#         f"Trainable Params (M): {trainable_params_m:.2f} | Trainable FLOPs (G): {trainable_flops_str}",
#         flush=True,
#     )
#     if trainable_params_m > 0:
#         print("[Approx] Trainable breakdown by LoRA target module:", flush=True)
#         for name in sorted(LORA_TARGET_MODULES, key=lambda n: trainable_module_params_m.get(n, 0.0), reverse=True):
#             module_params_m = trainable_module_params_m.get(name, 0.0)
#             if module_params_m == 0:
#                 continue
#             module_flops_g = trainable_module_flops_g.get(name)
#             module_flops_str = f"{module_flops_g:.4f}" if module_flops_g is not None else "N/A"
#             print(f"  {name}: Params (M): {module_params_m:.4f} | FLOPs (G): {module_flops_str}", flush=True)
#         if trainable_other_params_m > 0:
#             print(f"  ignored: Params (M): {trainable_other_params_m:.4f}", flush=True)

#     # Torch compile (best-effort)
#     if hasattr(torch, "compile") and device.type == "cuda":
#         try:
#             try:
#                 import torch._inductor as _inductor
#                 _inductor.config.triton.cudagraphs = False
#                 print("torch.compile: disabled cudagraphs for stability")
#             except Exception:
#                 pass
#             pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
#             pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=False)
#             print("torch.compile enabled for UNet/ControlNet")
#         except Exception as e:
#             print(f"[WARN] torch.compile failed: {e}")

#     t0 = time.time()
#     total_time = 0.0
#     processed = 0
#     for _size, group_paths in groups.items():
#         for start in range(0, len(group_paths), max(1, args.batch_size)):
#             batch_paths = group_paths[start:start + max(1, args.batch_size)]
#             try:
#                 imgs = []
#                 for p in batch_paths:
#                     img = Image.open(p).convert("RGB")
#                     if resize_scale != 1.0:
#                         img = img.resize(_size, resample=Image.BILINEAR)
#                     imgs.append(TF.to_tensor(img))
#                 x = torch.stack(imgs, dim=0).to(device)  # [B,3,H,W], [0,1]
#                 t = time.perf_counter()

#                 if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
#                     torch.compiler.cudagraph_mark_step_begin()
#                 y = purify_tensor(
#                     x, pipe,
#                     canny_low=args.canny_low, canny_high=args.canny_high,
#                     num_steps=args.num_inference_step,
#                     strength=args.strength,
#                     guidance_scale=args.guidance_scale,
#                     control_scale=args.control_scale,
#                     seed=args.seed
#                 )
#                 elapsed = time.perf_counter() - t
#                 total_time += elapsed
#                 per_img_ms = (elapsed / len(batch_paths)) * 1000
#                 print(f"batch {len(batch_paths)} done in {elapsed:.5f}s ({per_img_ms:.2f} ms/img)")
#                 if processed == 0:
#                     latency_ms = per_img_ms
#                     print(
#                         f"Params (M): {params_m:.2f} | FLOPs (G): {flops_str} | "
#                         f"Trainable Params (M): {trainable_params_m:.2f} | "
#                         f"Trainable FLOPs (G): {trainable_flops_str} | Latency (ms): {latency_ms:.2f}"
#                     )

#                 # Save with same relative structure under output_dir
#                 for j, p in enumerate(batch_paths):
#                     rel = p.relative_to(in_dir)
#                     save_path = (out_dir / rel).with_suffix(".png")
#                     save_path.parent.mkdir(parents=True, exist_ok=True)
#                     TF.to_pil_image(y[j].clamp(0,1).cpu()).save(save_path)
#                     processed += 1
#                     print(f"[{processed}/{len(paths)}] saved -> {save_path}")
#             except Exception as e:
#                 for p in batch_paths:
#                     print(f"[ERROR] {p}: {e}")

#     print(f"done in {time.time()-t0:.5f}s  images={len(paths)}  out={out_dir}")
#     latency_ms = (total_time / len(paths) * 1000) if paths else 0.0
#     print(
#         f"Params (M): {params_m:.2f} | FLOPs (G): {flops_str} | "
#         f"Trainable Params (M): {trainable_params_m:.2f} | "
#         f"Trainable FLOPs (G): {trainable_flops_str} | Latency (ms): {latency_ms:.2f}"
#     )

# if __name__ == "__main__":
#     main()
