"""
Stable Zero123 SDS (Score Distillation Sampling) Guidance for SAM3D Training.

Uses Stable Zero123 (stabilityai/stable-zero123) to provide a view-conditioned
diffusion prior that guides Gaussian color optimization from novel viewpoints.

Based on PAD3r/threestudio's Stable Zero123 guidance implementation.

Checkpoint: https://huggingface.co/stabilityai/stable-zero123
"""

import importlib
import os

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from diffusers import DDIMScheduler
from omegaconf import OmegaConf


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    pl_sd = torch.load(ckpt, map_location=device)

    if "global_step" in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder for SDS (only encoding)
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


class StableZero123Guidance:
    def __init__(
        self,
        device="cuda",
        ckpt_path="extern/ldm_zero123/stable_zero123.ckpt",
        config_path="extern/ldm_zero123/sd-objaverse-finetune-c_concat-256.yaml",
        guidance_scale=3.0,
        min_step_ratio=0.02,
        max_step_ratio=0.5,
        cond_elevation_deg=0.0,
        cond_azimuth_deg=0.0,
        cond_distance=3.8,
        half_precision=True,
        grad_clip_val=None,
        prediction_type="epsilon",
        save_debug=False,
        keep_decoder=False,
        ddim_steps=5,
    ):
        self.device = device
        self.guidance_scale = guidance_scale
        self.cond_elevation_deg = cond_elevation_deg
        self.cond_azimuth_deg = cond_azimuth_deg
        self.cond_distance = cond_distance
        self.grad_clip_val = grad_clip_val
        self.prediction_type = prediction_type
        self.save_debug = save_debug
        self.ddim_steps = ddim_steps

        # Auto-download checkpoint from HuggingFace if not found locally
        if not os.path.exists(ckpt_path):
            print(f"[StableZero123] Checkpoint not found at {ckpt_path}, downloading from HuggingFace...")
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id="stabilityai/stable-zero123",
                filename="stable_zero123.ckpt",
            )
            print(f"[StableZero123] Downloaded to {ckpt_path}")

        print(f"[StableZero123] Loading model from {ckpt_path} ...")

        self.weights_dtype = torch.float16 if half_precision else torch.float32

        # Keep decoder if debug or precompute_ddim needs it
        need_decoder = save_debug or keep_decoder
        config = OmegaConf.load(config_path)
        self.model = load_model_from_config(
            config, ckpt_path, device=device, vram_O=not need_decoder
        )
        self.model.to(dtype=self.weights_dtype)

        # Keep CLIP LayerNorm modules in fp32 for stability
        def recursive_layernorm_fp32(module):
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(module, attr_name)
                except Exception:
                    continue
                if isinstance(attr, clip.model.LayerNorm):
                    attr.to(dtype=torch.float32)
                elif isinstance(attr, torch.nn.Sequential):
                    for sub in attr:
                        recursive_layernorm_fp32(sub)
                elif isinstance(attr, torch.nn.Module):
                    recursive_layernorm_fp32(attr)
        recursive_layernorm_fp32(self.model)

        # Keep the VAE decoder permanently in fp32 (fp16 decode causes a brightness
        # shift), so decode_latents doesn't have to cast the whole VAE every call
        if hasattr(self.model.first_stage_model, 'decoder'):
            self.model.first_stage_model.decoder.float()
            self.model.first_stage_model.post_quant_conv.float()

        for p in self.model.parameters():
            p.requires_grad_(False)

        # Timestep schedule
        self.num_train_timesteps = config.model.params.timesteps
        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            config.model.params.linear_start,
            config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step_ratio_start = min_step_ratio
        self.max_step_ratio_start = max_step_ratio
        self.min_step = int(self.num_train_timesteps * min_step_ratio)
        self.max_step = int(self.num_train_timesteps * max_step_ratio)

        self.alphas = self.scheduler.alphas_cumprod.to(device)

        # Precomputed reference embeddings (set by set_reference_image)
        self.c_crossattn = None
        self.c_concat = None

        # Debug buffer: last N denoised images
        self.debug_buffer = deque(maxlen=10)

        print(f"[StableZero123] Loaded. Steps: [{self.min_step}, {self.max_step}], "
              f"guidance_scale={guidance_scale}, prediction_type={prediction_type}, "
              f"decoder={'kept' if need_decoder else 'deleted'}")
        print(f"[StableZero123] Cond camera: elev={cond_elevation_deg}, "
              f"az={cond_azimuth_deg}, dist={cond_distance}")

    def update_steps(self, progress, min_step_ratio_end=None, max_step_ratio_end=None):
        """Linearly anneal min_step and max_step based on training progress.

        Args:
            progress: float in [0, 1], fraction of training completed.
            min_step_ratio_end: target min_step ratio at progress=1. None = no annealing.
            max_step_ratio_end: target max_step ratio at progress=1. None = no annealing.
        """
        if min_step_ratio_end is not None:
            ratio = self.min_step_ratio_start + progress * (min_step_ratio_end - self.min_step_ratio_start)
            self.min_step = int(self.num_train_timesteps * ratio)
        if max_step_ratio_end is not None:
            ratio = self.max_step_ratio_start + progress * (max_step_ratio_end - self.max_step_ratio_start)
            self.max_step = int(self.num_train_timesteps * ratio)
        # Ensure min <= max
        self.min_step = min(self.min_step, self.max_step)

    @torch.no_grad()
    def set_reference_image(self, image_tensor):
        """
        Precompute reference embeddings from a single GT reference image.

        Args:
            image_tensor: (1, 3, 256, 256) in [0, 1], RGB
        """
        img = image_tensor.to(self.device, dtype=self.weights_dtype)
        img = img * 2.0 - 1.0  # [0,1] -> [-1,1]

        self.c_crossattn = self.model.get_learned_conditioning(img).to(self.weights_dtype)
        self.c_concat = self.model.encode_first_stage(img).mode()

        print(f"[StableZero123] Reference embeddings: "
              f"c_crossattn={self.c_crossattn.shape}, c_concat={self.c_concat.shape}")

    # Keep old name as alias for backward compatibility
    get_img_embeds = set_reference_image

    @torch.no_grad()
    def precompute_reference_embeddings(self, frame_images):
        """
        Precompute reference embeddings for all frames.

        Args:
            frame_images: dict mapping frame_idx -> (1, 3, 256, 256) tensor in [0, 1]
        """
        self._frame_crossattn = {}
        self._frame_concat = {}
        for frame_idx, image_tensor in frame_images.items():
            img = image_tensor.to(self.device, dtype=self.weights_dtype)
            img = img * 2.0 - 1.0
            self._frame_crossattn[frame_idx] = self.model.get_learned_conditioning(img).to(self.weights_dtype)
            self._frame_concat[frame_idx] = self.model.encode_first_stage(img).mode()
        print(f"[StableZero123] Precomputed reference embeddings for {len(frame_images)} frames")

    def set_active_frame(self, frame_idx):
        """Set the active reference frame for subsequent sds_loss calls."""
        self.c_crossattn = self._frame_crossattn[frame_idx]
        self.c_concat = self._frame_concat[frame_idx]

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs):
        """
        Differentiable VAE encode for rendered images.

        Args:
            imgs: (B, 3, H, W) in [0, 1]

        Returns:
            latents: (B, 4, H/8, W/8) scaled
        """
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images. The decoder is kept permanently in fp32
        (fp16 decode causes a brightness shift)."""
        # Unscale latents (matching decode_first_stage internals)
        z = 1.0 / self.model.scale_factor * latents.float()
        image = self.model.first_stage_model.decode(z)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond(self, elevation_deg, azimuth_deg, distance):
        """
        Build conditioning dict for the UNet.

        The conditioning uses the precomputed GT reference image embeddings
        (c_crossattn for CLIP cross-attention, c_concat for VAE latent concat).

        Camera embedding uses PAD3r/threestudio convention:
        T = [polar_angle, sin(relative_azimuth), cos(relative_azimuth), cond_polar]

        Args:
            elevation_deg: float, target elevation in degrees
            azimuth_deg: float, target azimuth in degrees
            distance: float, target camera distance (unused by Stable Zero123 but kept for API)
        """
        elevation = torch.tensor([elevation_deg], device=self.device, dtype=torch.float32)
        azimuth = torch.tensor([azimuth_deg], device=self.device, dtype=torch.float32)

        T = torch.stack(
            [
                torch.deg2rad((90 - elevation) - (90 - self.cond_elevation_deg)),
                torch.sin(torch.deg2rad(azimuth - self.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth - self.cond_azimuth_deg)),
                torch.deg2rad(90 - torch.full_like(elevation, self.cond_elevation_deg)),
            ],
            dim=-1,
        )[:, None, :].to(self.device, dtype=self.weights_dtype)  # (1, 1, 4)

        clip_emb = self.model.cc_projection(
            torch.cat([self.c_crossattn.repeat(len(T), 1, 1), T], dim=-1)
        )

        cond = {}
        cond["c_crossattn"] = [
            torch.cat([torch.zeros_like(clip_emb), clip_emb], dim=0)
        ]
        cond["c_concat"] = [
            torch.cat(
                [
                    torch.zeros_like(self.c_concat).repeat(len(T), 1, 1, 1),
                    self.c_concat.repeat(len(T), 1, 1, 1),
                ],
                dim=0,
            )
        ]
        return cond

    @torch.no_grad()
    def _ddim_denoise(self, latents_noisy, t, cond, ddim_steps=None):
        """
        Run DDIM denoising from latents_noisy at timestep t with CFG.

        Supports both epsilon-prediction and v-prediction modes (set via self.prediction_type).
        - epsilon: x0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
        - v_prediction: x0 = sqrt(alpha) * x_t - sqrt(1-alpha) * v  (no division, more stable)

        Returns:
            ddim_rgb: (1, 3, 256, 256) decoded RGB in [0, 1], or None if decoder unavailable
        """
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        if not hasattr(self.model.first_stage_model, 'decoder'):
            return None

        t_val = t.item()
        ddim_timesteps = torch.linspace(t_val, 0, ddim_steps + 1).long().to(self.device)

        x_t = latents_noisy.clone()
        for i in range(ddim_steps):
            t_cur = ddim_timesteps[i].unsqueeze(0)
            t_next = ddim_timesteps[i + 1].unsqueeze(0)

            # CFG: run both unconditional and conditional
            x_in_ddim = torch.cat([x_t] * 2)
            t_in_ddim = torch.cat([t_cur] * 2)
            model_out = self.model.apply_model(x_in_ddim.to(self.weights_dtype), t_in_ddim, cond)
            out_uncond, out_cond = model_out.chunk(2)
            model_output = out_uncond + self.guidance_scale * (out_cond - out_uncond)

            alpha_cur = self.alphas[t_cur].reshape(-1, 1, 1, 1)
            alpha_next = self.alphas[t_next].reshape(-1, 1, 1, 1)

            if self.prediction_type == "v_prediction":
                # v-prediction: model output treated as v = sqrt(alpha)*eps - sqrt(1-alpha)*x0
                x0_pred = torch.sqrt(alpha_cur) * x_t - torch.sqrt(1 - alpha_cur) * model_output
                eps_pred = torch.sqrt(alpha_cur) * model_output + torch.sqrt(1 - alpha_cur) * x_t
            else:
                # epsilon-prediction (default)
                x0_pred = (x_t - torch.sqrt(1 - alpha_cur) * model_output) / torch.sqrt(alpha_cur)
                eps_pred = model_output

            x_t = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * eps_pred

        decoded = self.decode_latents(x_t)
        return decoded

    @torch.no_grad()
    def ddim_target(self, rendered_rgb, elevation_deg, azimuth_deg, distance):
        """
        Run DDIM denoising on a rendered image and return the denoised RGB target.

        Fully under torch.no_grad() — the returned image is detached and safe
        to use as an L2 target (gradients only flow through the GS render side).

        Args:
            rendered_rgb: (H, W, 3) in [0, 255] — rendered GS from novel view
            elevation_deg: target camera elevation in degrees
            azimuth_deg: target camera azimuth in degrees
            distance: target camera distance

        Returns:
            tuple of (ddim_rgb, w) where ddim_rgb is (H, W, 3) float tensor in [0, 1]
            and w is the scalar timestep weight (1 - alpha_bar_t), or (None, None).
        """
        assert self.c_crossattn is not None, "Call set_active_frame first"

        # Encode rendered image to latent space
        img = rendered_rgb / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_images(img)  # (1, 4, 32, 32)

        # Build conditioning
        cond = self.get_cond(elevation_deg, azimuth_deg, distance)

        # Sample random timestep
        t = torch.randint(
            self.min_step, self.max_step + 1,
            [1], dtype=torch.long, device=self.device,
        )

        # Timestep weight: w(t) = 1 - alpha_bar_t
        w = (1 - self.alphas[t]).item()

        # Add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # DDIM denoise
        ddim_decoded = self._ddim_denoise(latents_noisy, t, cond)
        if ddim_decoded is None:
            return None, None

        # (1, 3, 256, 256) -> (256, 256, 3)
        return ddim_decoded[0].permute(1, 2, 0).clamp(0, 1), w

    def sds_loss(self, rendered_rgb, elevation_deg, azimuth_deg, distance, return_ddim=False):
        """
        Compute SDS loss using Zero123 view-conditioned diffusion.

        Reference image (GT video frame) is set via set_reference_image() and used
        as conditioning (c_crossattn + c_concat). The rendered_rgb is the current
        GS rendering from a novel view — it gets encoded to latent space, noise is
        added, and the UNet predicts the noise conditioned on the GT reference.

        Args:
            rendered_rgb: (H, W, 3) in [0, 255] — rendered GS from novel view (noisy latent input)
            elevation_deg: target camera elevation in degrees (relative to reference)
            azimuth_deg: target camera azimuth in degrees (relative to reference)
            distance: target camera distance
            return_ddim: if True, also return DDIM-denoised image as uint8 numpy (256, 256, 3)

        Returns:
            loss: scalar SDS loss (if return_ddim=False)
            (loss, ddim_np): tuple of loss and DDIM image (if return_ddim=True)
        """
        assert self.c_crossattn is not None, "Call set_reference_image first"
        batch_size = 1

        # Encode rendered image to latent space (differentiable — this is the "noisy latent" input)
        img = rendered_rgb / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_images(img)  # (1, 4, 32, 32)

        # Build conditioning from GT reference image embeddings + camera pose
        cond = self.get_cond(elevation_deg, azimuth_deg, distance)

        # Sample random timestep
        t = torch.randint(
            self.min_step, self.max_step + 1,
            [batch_size], dtype=torch.long, device=self.device,
        )

        # Add noise to rendered image latents
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # UNet predicts noise, conditioned on GT reference (c_concat + c_crossattn)
            x_in = torch.cat([latents_noisy] * 2)  # doubled for CFG
            t_in = torch.cat([t] * 2)
            noise_pred = self.model.apply_model(
                x_in.to(self.weights_dtype), t_in, cond
            )

        # Classifier-free guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # DDIM denoising for precompute or debug
        ddim_np = None
        if return_ddim or self.save_debug:
            with torch.no_grad():
                ddim_decoded = self._ddim_denoise(latents_noisy, t, cond)
                if ddim_decoded is not None:
                    ddim_np = (ddim_decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Debug visualization buffer
        if self.save_debug:
            with torch.no_grad():
                rendered_np = (img[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                denoised_np = None
                if hasattr(self.model.first_stage_model, 'decoder'):
                    alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
                    if self.prediction_type == "v_prediction":
                        pred_x0 = torch.sqrt(alpha_t) * latents_noisy - torch.sqrt(1 - alpha_t) * noise_pred
                    else:
                        pred_x0 = (latents_noisy - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    decoded = self.decode_latents(pred_x0)
                    denoised_np = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                self.debug_buffer.append({
                    'rendered': rendered_np,
                    'denoised': denoised_np,
                    'ddim': ddim_np,
                    'elevation': elevation_deg,
                    'azimuth': azimuth_deg,
                    't': t.item(),
                })

        # w(t) = (1 - alpha_bar_t) weighting
        w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        grad = grad.to(latents.dtype)

        # Optional gradient clipping
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # Loss via target formulation
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        if return_ddim:
            return loss, ddim_np
        return loss
