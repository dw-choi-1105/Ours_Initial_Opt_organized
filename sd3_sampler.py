from typing import List, Tuple, Optional
import math
import torch
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
import numpy as np
import torch.nn.functional as F
from custom_util import *
from diffusers import AutoencoderTiny
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image
from torch.cuda.amp import autocast

# =======================================================================
# Factory
# =======================================================================
__SOLVER__ = {}

def register_solver(name:str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name:str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

import time

def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _tic():
    _sync(); return time.perf_counter()

def _toc(t0):
    _sync(); return time.perf_counter() - t0

class StableDiffusion3Base():
    def __init__(
        self,
        model_key: str = "stabilityai/stable-diffusion-3.5-medium",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        keep_pipe_for_encoding: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
        )
        
        self.scheduler = pipe.scheduler
        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3
        
        # Use self.dtype instead of hardcoded torch.float16
        self.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd3", torch_dtype=self.dtype
        ).to(self.device).eval()
        
        self.transformer = pipe.transformer.to(self.device).eval()
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            if hasattr(self.transformer, 'enable_gradient_checkpointing'):
                self.transformer.enable_gradient_checkpointing()
                print("✓ Gradient checkpointing enabled for transformer")
            else:
                print("⚠ Gradient checkpointing not available for this model")
        
        self.transformer.requires_grad_(False)
        
        if keep_pipe_for_encoding:
            pipe.transformer = None
            pipe.vae = None
            self._encode_prompt = pipe.encode_prompt
            self._pipe_for_encoding = pipe
        else:
            self._encode_prompt = None
            self._pipe_for_encoding = None
            del pipe
        
        self.seed = None
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None else 8
        )
    
    def encode_prompt(
        self,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
        prompt_3: Optional[List[str]] = None,
        negative_prompt_3: Optional[List[str]] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        prompt_embeds, neg_embeds, pooled, neg_pooled = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=torch.device(self.device),
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            clip_skip=clip_skip,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        return prompt_embeds.to(self.dtype), pooled.to(self.dtype)
    
    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):
        H, W = img_size
        lH, lW = H//self.vae_scale_factor, W//self.vae_scale_factor
        lC = self.transformer.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)
        
        if self.seed is not None:
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype, generator=torch.Generator(self.device).manual_seed(42))
        else:
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
        return z
    
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        img_latent = self.vae.encode(image, return_dict=False)[0]
        if hasattr(img_latent, "sample"):
            img_latent = img_latent.sample()
        img_latent = (img_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return img_latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        img = self.vae.decode(z / self.vae.config.scaling_factor + self.vae.config.shift_factor, return_dict=False)[0]
        return img
    
    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        v = self.transformer(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False)[0]
        return v

@register_solver("naive")
class SD3Euler(StableDiffusion3Base):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3.5-medium', 
                 device='cuda', dtype: torch.dtype = torch.float32,
                 use_gradient_checkpointing: bool = False):
        super().__init__(model_key=model_key, device=device, dtype=dtype,
                        use_gradient_checkpointing=use_gradient_checkpointing)
    
    def sample(self, prompts: List[str], NFE:int, img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        
        # encode text prompts
        with torch.no_grad():
            if prompt_emb is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_emb[0], prompt_emb[1]
            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)
            
            if null_emb is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_emb[0], null_emb[1]
            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)
        
        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent
        
        # timesteps (default option. You can make your custom here.)
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE+2, device=self.transformer.device)
        timesteps = self.scheduler.timesteps
        timesteps = timesteps[1:]
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        
        # Solve ODE
        images = []
        pbar = tqdm(timesteps[:-1], total=NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0
            
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            
            pred_v_fin = pred_null_v + cfg_scale * (pred_v - pred_null_v)
            z0t = z - sigma * pred_v_fin
            z = z + (sigma_next - sigma) * pred_v_fin
            
            img_out = self.decode(z0t).float()
            img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            images.append(img_out)
        
        # decode
        with torch.no_grad():
            img = self.decode(z)
        
        return img, images
    
    # ──────────────────────────────────────────────────────────
    # FireFlow sampling (MAXIMUM MEMORY OPTIMIZATION)
    # ──────────────────────────────────────────────────────────
    def fireflow_sample(
        self,
        initial_z: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
        prompt_emb: torch.Tensor,
        pooled_emb: torch.Tensor,
        null_emb: torch.Tensor,
        null_pooled: torch.Tensor,
        cfg_scale: float = 1.0,
        return_z0t_at_step: int = -1,
    ):
        """
        Maximum memory-optimized FireFlow sampling.
        
        Optimizations applied:
        1. In-place operations
        2. Tensor buffer reuse
        3. Gradient checkpointing (if enabled)
        4. Explicit tensor deletion
        5. Periodic cache clearing
        """
        z = initial_z.to(dtype=self.dtype)
        hat_velocity = None
        
        # Pre-allocate timestep tensors
        batch_size = z.shape[0]
        ts_buffer = torch.empty(batch_size, device=self.device, dtype=timesteps.dtype)
        # images = []

        for step_i in range(len(timesteps) - 1):
            t_curr = timesteps[step_i]
            t_next = timesteps[step_i + 1]
            sigma_curr = sigmas[step_i]
            sigma_next = sigmas[step_i + 1]
            dt = sigma_next - sigma_curr
            
            # --- velocity at current position ---
            if hat_velocity is None:
                ts_buffer.fill_(t_curr)
                velocity = self.predict_vector(z, ts_buffer, prompt_emb, pooled_emb)
                
                if cfg_scale != 1.0:
                    vel_null = self.predict_vector(z, ts_buffer, null_emb, null_pooled)
                    velocity = vel_null + cfg_scale * (velocity - vel_null)
                    del vel_null
            else:
                velocity = hat_velocity
            
            # --- midpoint ---
            z_mid = z.add(velocity, alpha=dt / 2)
            
            t_mid = t_curr + (t_next - t_curr) / 2
            ts_buffer.fill_(t_mid)
            
            vel_mid = self.predict_vector(z_mid, ts_buffer, prompt_emb, pooled_emb)
            
            if cfg_scale != 1.0:
                vel_mid_null = self.predict_vector(z_mid, ts_buffer, null_emb, null_pooled)
                vel_mid = vel_mid_null + cfg_scale * (vel_mid - vel_mid_null)
                del vel_mid_null
            
            del z_mid
            hat_velocity = vel_mid
            
            # --- update using midpoint velocity ---
            # z0t = z - sigma_curr * vel_mid
            # img_out = self.decode(z0t).float()
            # img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            # images.append(img_out)

            z = z.add(vel_mid, alpha=dt)

            # Periodic cleanup
            if step_i % 2 == 0:
                torch.cuda.empty_cache()
        
        # return z, images
        return z

    @torch.no_grad()
    def fireflow_sample_w_process(
        self,
        initial_z: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
        prompt_emb: torch.Tensor,
        pooled_emb: torch.Tensor,
        null_emb: torch.Tensor,
        null_pooled: torch.Tensor,
        cfg_scale: float = 1.0,
        return_z0t_at_step: int = -1,
    ):
        """
        Maximum memory-optimized FireFlow sampling.
        
        Optimizations applied:
        1. In-place operations
        2. Tensor buffer reuse
        3. Gradient checkpointing (if enabled)
        4. Explicit tensor deletion
        5. Periodic cache clearing
        """
        z = initial_z.to(dtype=self.dtype)
        hat_velocity = None
        
        # Pre-allocate timestep tensors
        batch_size = z.shape[0]
        ts_buffer = torch.empty(batch_size, device=self.device, dtype=timesteps.dtype)
        images = []

        for step_i in range(len(timesteps) - 1):
            t_curr = timesteps[step_i]
            t_next = timesteps[step_i + 1]
            sigma_curr = sigmas[step_i]
            sigma_next = sigmas[step_i + 1]
            dt = sigma_next - sigma_curr
            
            # --- velocity at current position ---
            if hat_velocity is None:
                ts_buffer.fill_(t_curr)
                velocity = self.predict_vector(z, ts_buffer, prompt_emb, pooled_emb)
                
                if cfg_scale != 1.0:
                    vel_null = self.predict_vector(z, ts_buffer, null_emb, null_pooled)
                    velocity = vel_null + cfg_scale * (velocity - vel_null)
                    del vel_null
            else:
                velocity = hat_velocity
            
            # --- midpoint ---
            z_mid = z.add(velocity, alpha=dt / 2)
            
            t_mid = t_curr + (t_next - t_curr) / 2
            ts_buffer.fill_(t_mid)
            
            vel_mid = self.predict_vector(z_mid, ts_buffer, prompt_emb, pooled_emb)
            
            if cfg_scale != 1.0:
                vel_mid_null = self.predict_vector(z_mid, ts_buffer, null_emb, null_pooled)
                vel_mid = vel_mid_null + cfg_scale * (vel_mid - vel_mid_null)
                del vel_mid_null
            
            del z_mid
            hat_velocity = vel_mid
            
            # --- store process ---
            z0t = z - sigma_curr * vel_mid
            img_out = self.decode(z0t).float()
            img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            images.append(img_out)

            # --- update using midpoint velocity ---
            z = z.add(vel_mid, alpha=dt)

            # Periodic cleanup
            if step_i % 2 == 0:
                torch.cuda.empty_cache()
        
        return z, images
    
    # ──────────────────────────────────────────────────────────
    # Euler sampling
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def euler_sample(
        self,
        initial_z: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
        prompt_emb: torch.Tensor,
        pooled_emb: torch.Tensor,
        null_emb: torch.Tensor,
        null_pooled: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        z = initial_z.clone().to(dtype=self.dtype)
        batch_size = z.shape[0]
        ts_buffer = torch.empty(batch_size, device=self.device, dtype=timesteps.dtype)
        
        images = []
        for step_i in range(len(timesteps) - 1):
            t_curr = timesteps[step_i]
            ts_buffer.fill_(t_curr)
            
            pred_v = self.predict_vector(z, ts_buffer, prompt_emb, pooled_emb)
            
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, ts_buffer, null_emb, null_pooled)
                pred_v = pred_null_v + cfg_scale * (pred_v - pred_null_v)
                del pred_null_v
            
            sigma      = sigmas[step_i]
            sigma_next = sigmas[step_i + 1]

            z0t = z - sigma * pred_v
            img_out = self.decode(z0t).float()
            img_out = (img_out / 2 + 0.5).clamp(0, 1).detach().cpu()
            images.append(img_out)

            z = z.add(pred_v, alpha=sigma_next - sigma)
            
            if step_i % 5 == 0:
                torch.cuda.empty_cache()
        
        img = self.decode(z)
        return img, images
    
    def euler_sample_wo_process(
        self,
        initial_z: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
        prompt_emb: torch.Tensor,
        pooled_emb: torch.Tensor,
        null_emb: torch.Tensor,
        null_pooled: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        z = initial_z.clone().to(dtype=self.dtype)
        batch_size = z.shape[0]
        ts_buffer = torch.empty(batch_size, device=self.device, dtype=timesteps.dtype)
        
        for step_i in range(len(timesteps) - 1):
            t_curr = timesteps[step_i]
            ts_buffer.fill_(t_curr)
            
            pred_v = self.predict_vector(z, ts_buffer, prompt_emb, pooled_emb)
            
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, ts_buffer, null_emb, null_pooled)
                pred_v = pred_null_v + cfg_scale * (pred_v - pred_null_v)
                del pred_null_v
            
            sigma      = sigmas[step_i]
            sigma_next = sigmas[step_i + 1]

            z = z.add(pred_v, alpha=sigma_next - sigma)
            
            if step_i % 2 == 0:
                torch.cuda.empty_cache()
        
        return z

    # ──────────────────────────────────────────────────────────
    # Data-consistency loss (MAXIMUM MEMORY OPTIMIZATION)
    # ──────────────────────────────────────────────────────────
    def compute_data_consistency_loss(
        self, z0t, A, y,
        sigma_val: float,
        noise_std: float = 0.03,
        phi: float = 1.0,
        eta_tilde: float = 0.8,
        eta_min: float = 1e-4,
    ) -> torch.Tensor:
        """
        Maximum memory-optimized data consistency loss.
        
        Key optimizations:
        1. Minimal intermediate tensor storage
        2. Direct computation of loss terms
        3. Explicit cleanup
        """
        
        device = z0t.device
        sigma_t  = torch.tensor(sigma_val, device=device, dtype=torch.float32)
        lambda_t = ((1 - sigma_t).clamp(0, 1)) ** phi
        wBP = (1 - lambda_t)
        wLS = lambda_t
        eta_reg = max(eta_min, (noise_std ** 2) * eta_tilde)
        
        # Decode
        x = self.decode(z0t)
        
        if x.dtype != torch.float32:
            x = x.float()
        
        x_flat = x.reshape(x.size(0), -1).contiguous()
        y_flat = y.reshape(y.size(0), -1).contiguous()
        
        # Compute Ax once
        Ax = A.A(x_flat)
        loss_LS = torch.linalg.norm(Ax - y_flat)
        loss_BP = torch.linalg.norm(
            A.A_pinv_add_eta(Ax, eta_reg) - A.A_pinv_add_eta(y_flat, eta_reg)
        )
        
        del Ax, x_flat
        
        # return wBP * loss_BP + wLS * loss_LS
        return loss_BP

    # Frequency-Decomposed Guidance  
    def compute_data_consistency_loss_FAHG(
        self, z0t, A, y,
        sigma_val: float,
        noise_std: float = 0.03,
        phi: float = 1.0,
        eta_tilde: float = 0.8,
        eta_min: float = 1e-4,
        freq_split_ratio: float = 0.3,
        save_debug_images: bool = False,
        debug_save_path: str = "./debug_freq_images",
    ) -> torch.Tensor:
        
        device = z0t.device
        sigma_t = torch.tensor(sigma_val, device=device, dtype=torch.float32)
        lambda_t = ((1 - sigma_t).clamp(0, 1)) ** phi
        eta_reg = max(eta_min, (noise_std ** 2) * eta_tilde)
        
        # Decode
        x = self.decode(z0t)
        if x.dtype != torch.float32:
            x = x.float()
        
        # ===== Ensure both x and y are 4D =====
        if x.dim() == 2:
            if y.dim() == 4:
                B, C, H, W = y.shape
                x = x.view(B, C, H, W)
            elif y.dim() == 2:
                B = x.size(0)
                total_pixels = y.size(-1)
                num_channels = 3
                spatial_size = int((total_pixels / num_channels) ** 0.5)
                x = x.view(B, num_channels, spatial_size, spatial_size)
                y = y.view(B, num_channels, spatial_size, spatial_size)
            else:
                raise ValueError(f"Unexpected y shape: {y.shape}")
        
        if y.dim() == 2:
            B, C, H, W = x.shape
            y = y.view(B, C, H, W)
        
        B, C, H, W = x.shape
        assert y.shape == x.shape, f"Shape mismatch: x={x.shape}, y={y.shape}"
        
        x_flat = x.reshape(B, -1).contiguous()
        y_flat = y.reshape(B, -1).contiguous()
        
        # Compute Ax
        Ax = A.A(x_flat)
        Ax_spatial = Ax.view(B, C, H, W)
        
        # FFT
        Ax_freq = torch.fft.rfft2(Ax_spatial, dim=(-2, -1))
        y_freq = torch.fft.rfft2(y, dim=(-2, -1))
        
        # Frequency dimensions
        freq_H = Ax_freq.shape[-2]
        freq_W = Ax_freq.shape[-1]
        
        cutoff_h = int(freq_H * freq_split_ratio)
        cutoff_w = int(freq_W * freq_split_ratio)
        
        # Create masks
        lf_mask = torch.zeros_like(Ax_freq, dtype=torch.float32)
        lf_mask[..., :cutoff_h, :cutoff_w] = 1.0
        hf_mask = 1.0 - lf_mask
        
        # ===== Low-frequency components =====
        # Extract LF in frequency domain
        Ax_freq_lf = Ax_freq * lf_mask
        y_freq_lf = y_freq * lf_mask
        
        # Convert back to SPATIAL domain
        Ax_spatial_lf = torch.fft.irfft2(Ax_freq_lf, s=(H, W), dim=(-2, -1))
        y_spatial_lf = torch.fft.irfft2(y_freq_lf, s=(H, W), dim=(-2, -1))
        
        # ===== Save Low-Frequency Images =====
        if save_debug_images:
            import os
            from torchvision.utils import save_image
            
            os.makedirs(debug_save_path, exist_ok=True)
            
            # Normalize to [0, 1] for saving
            def normalize_for_save(tensor):
                # tensor: [B, C, H, W]
                tensor_min = tensor.min()
                tensor_max = tensor.max()
                if tensor_max > tensor_min:
                    return (tensor - tensor_min) / (tensor_max - tensor_min)
                return tensor
            
            # Save LF images
            Ax_lf_norm = normalize_for_save(Ax_spatial_lf)
            y_lf_norm = normalize_for_save(y_spatial_lf)
            
            save_image(Ax_lf_norm, os.path.join(debug_save_path, 'Ax_spatial_lf.png'))
            save_image(y_lf_norm, os.path.join(debug_save_path, 'y_spatial_lf.png'))
        
        # Compute LF loss in SPATIAL domain (LS guidance)
        loss_LF = torch.linalg.norm(Ax_spatial_lf - y_spatial_lf)
        
        # ===== High-frequency components =====
        # Extract HF in frequency domain
        Ax_freq_hf = Ax_freq * hf_mask
        y_freq_hf = y_freq * hf_mask
        
        # Convert to spatial domain
        hf_Ax_spatial = torch.fft.irfft2(Ax_freq_hf, s=(H, W), dim=(-2, -1))
        hf_y_spatial = torch.fft.irfft2(y_freq_hf, s=(H, W), dim=(-2, -1))
        
        # ===== Save High-Frequency Images =====
        if save_debug_images:
            # Save HF images
            hf_Ax_norm = normalize_for_save(hf_Ax_spatial)
            hf_y_norm = normalize_for_save(hf_y_spatial)
            
            save_image(hf_Ax_norm, os.path.join(debug_save_path, 'hf_Ax_spatial.png'))
            save_image(hf_y_norm, os.path.join(debug_save_path, 'hf_y_spatial.png'))
            
            # Also save original images for comparison
            Ax_norm = normalize_for_save(Ax_spatial)
            y_norm = normalize_for_save(y)
            x_norm = normalize_for_save(x)
            
            save_image(Ax_norm, os.path.join(debug_save_path, 'Ax_spatial_full.png'))
            save_image(y_norm, os.path.join(debug_save_path, 'y_full.png'))
            save_image(x_norm, os.path.join(debug_save_path, 'x_decoded.png'))
            
            print(f"[DEBUG] Saved frequency decomposition images to {debug_save_path}")
        
        # Flatten for pseudo-inverse (BP guidance)
        hf_Ax_flat = hf_Ax_spatial.reshape(B, -1).contiguous()
        hf_y_flat = hf_y_spatial.reshape(B, -1).contiguous()
        
        # Compute HF loss in SPATIAL domain (BP guidance)
        loss_HF = torch.linalg.norm(
            A.A_pinv_add_eta(hf_Ax_flat, eta_reg) - 
            A.A_pinv_add_eta(hf_y_flat, eta_reg)
        )
        
        # Adaptive weighting
        alpha = lambda_t
        total_loss = (1 - alpha) * loss_LF + alpha * loss_HF

        return total_loss
        
    def data_consistency_DDPG_v2(
        self, z0t, A, y, sigma,
        step_scale=1.0, sigma_y=0.03, eta_tilde=0.8, eta_min=1e-4, phi=1.0, inner_steps=3
    ):
        lambda_t = ((1 - sigma).clamp(0, 1))**phi
        wBP, wLS = (1 - lambda_t), lambda_t
        eta_reg = max(eta_min, (sigma_y**2) * eta_tilde)
        
        z = z0t.detach().to(self.vae.device).requires_grad_(True)
        decay = float(sigma)**2
        step  = (step_scale * (0.25 + 0.75 * decay)) / inner_steps
        
        for _ in range(inner_steps):
            x = self.decode(z).float()
            
            x_flat = x.reshape(x.size(0), -1).contiguous()
            y_flat = y.reshape(y.size(0), -1).contiguous()
            
            Ax = A.A(x_flat)
            loss_BP = torch.linalg.norm(
                A.A_pinv_add_eta(Ax, eta_reg) - A.A_pinv_add_eta(y_flat, eta_reg)
            )
            loss_LS = torch.linalg.norm(Ax - y_flat)
            loss = wBP * loss_BP + wLS * loss_LS
            
            z_grad = torch.autograd.grad(loss, z)[0].to(dtype=self.dtype)
            z = (z - step * z_grad).detach().requires_grad_(True)
            
            del x, x_flat, Ax, loss_BP, loss_LS, loss, z_grad
            torch.cuda.empty_cache()
        
        z = z.detach().to(device=self.transformer.device, dtype=z0t.dtype)
        return z
    
    # 1) linear
    def lerp(self, a, b, t):
        return a + (b - a) * t
    # 2) exponential (normalized)
    def phi_exp(self, u, alpha=5.0):
        return (np.exp(alpha * u) - 1.0) / (np.exp(alpha) - 1.0)
    # 3) logarithmic (normalized)
    def phi_log(self, u, alpha=9.0):
        return np.log1p(alpha * u) / np.log1p(alpha)

    def posterior_sampling(
        self, 
        measurement, 
        operator, 
        task,
        prompts: List[str], NFE:int,
        img_shape: Optional[Tuple[int]]=None,
        cfg_scale: float=1.0, batch_size: int = 1,
        latent:Optional[List[torch.Tensor]]=None,
        prompt_embs:Optional[List[torch.Tensor]]=None,
        null_embs:Optional[List[torch.Tensor]]=None,
        step_scale_ps_1=None,
        step_scale_ps_2=None,
        inner_steps=None,
        sigma_y=None,
        stochasticity_weight=None,
        function_dc=None,
    ):
        A_funcs = operator
        y = measurement
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            if prompt_embs is None:
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            else:
                prompt_emb, pooled_emb = prompt_embs[0], prompt_embs[1]
            prompt_emb.to(self.transformer.device)
            pooled_emb.to(self.transformer.device)
            if null_embs is None:
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)
            else:
                null_prompt_emb, null_pooled_emb = null_embs[0], null_embs[1]
            null_prompt_emb.to(self.transformer.device)
            null_pooled_emb.to(self.transformer.device)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE+1, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        images_x0t = []
        pbar = tqdm(range(NFE), desc='SD3.5-FlowDPS')
        for i in pbar:
            t = timesteps[i]
            u = i / (NFE - 1)  # normalize to [0,1]
            if function_dc == 'linear':
                step_scale = self.lerp(step_scale_ps_1, step_scale_ps_2, u)
            elif function_dc == 'exponential':
                alpha_exp = 6.0
                t_exp = self.phi_exp(u, alpha_exp)
                step_scale = self.lerp(step_scale_ps_1, step_scale_ps_2, t_exp)
            elif function_dc == 'logarithm':
                alpha_log = 6.0
                t_log = self.phi_log(u, alpha_log)
                step_scale = self.lerp(step_scale_ps_1, step_scale_ps_2, t_log)
            elif function_dc == 'constant':
                step_scale = (step_scale_ps_1 + step_scale_ps_2) / 2
            
            timestep = t.expand(z.shape[0]).to(self.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0
            
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            delta = sigma - sigma_next
            pred_v_fin = pred_null_v + cfg_scale * (pred_v-pred_null_v)
            
            # denoising
            # 1. reverse process
            z_curr = z
            z_next = z - delta * pred_v_fin
            z0t = z - sigma * pred_v_fin
            z1t = z + (1-sigma) * pred_v_fin

            if i < NFE:
                z0y = self.data_consistency_DDPG_v2(
                        z0t, A_funcs, y, sigma,
                        step_scale=step_scale, 
                        sigma_y=sigma_y, 
                        eta_tilde=0.8, 
                        phi=1.0,
                        inner_steps=inner_steps
                    )
            else:
                z0y = z0t

            eps = torch.randn_like(z1t)
            sigma_next = sigmas[i+1]

            # flowdps stochasticity schedule
            if i < 14:
                alpha = 0.0
            else:
                alpha = (1 - sigma_next) ** 0.5 * stochasticity_weight
                if alpha > 1.0:
                    alpha = 1.0

            z1y = ((1-alpha**2)**0.5) * z1t + alpha * eps            
            z = (1 - sigma_next) * z0y + sigma_next * z1y

            img_out_x0t = self.decode(z0t).float()
            img_out_x0t = (img_out_x0t / 2 + 0.5).clamp(0, 1).detach().cpu()
            images_x0t.append(img_out_x0t)

        # decode
        with torch.no_grad():
            img = self.decode(z)
            img = (img / 2 + 0.5).clamp(0, 1).detach().cpu()
        return img, images_x0t