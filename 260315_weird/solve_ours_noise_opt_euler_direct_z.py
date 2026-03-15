"""
Direct z_init Gradient Optimization
Directly optimizes z_init as nn.Parameter (no Cayley / MoHE parameterization)
WITH LATENT NOISE PROGRESS VISUALIZATION AND POSTERIOR SAMPLING
OPTIMIZATION USES EULER SAMPLING (NOT FIREFLOW)
"""
import argparse
from pathlib import Path
from typing import List
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torchvision import transforms
from util import set_seed, get_img_list, process_text
# from sd3_sampler import get_solver
from sd3_sampler_sdo import get_solver
from torchvision.utils import make_grid
from custom_util import *
import matplotlib.pyplot as plt
import math
import numpy as np
import gc
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from debug_util import debug_save
import mlflow



@torch.no_grad
def precompute(args, prompts:List[str], solver) -> List[torch.Tensor]:
    prompt_emb_set = []
    pooled_emb_set = []
    num_samples = args.num_samples if args.num_samples > 0 else len(prompts)
    for prompt in prompts[:num_samples]:
        prompt_emb, pooled_emb = solver.encode_prompt(prompt)
        prompt_emb_set.append(prompt_emb)
        pooled_emb_set.append(pooled_emb)
    return prompt_emb_set, pooled_emb_set

def run(args):
    # ══════════════════════════════════════════════════════════════
    # MLflow setup
    # ══════════════════════════════════════════════════════════════
    mlflow.set_experiment(args.task)
    mlflow.start_run(run_name=args.workdir.name)
    mlflow.log_params({
        "seed": args.seed,
        "NFE": args.NFE,
        "inner_NFE": args.inner_NFE,
        "cfg_scale": args.cfg_scale,
        "img_size": args.img_size,
        "task": args.task,
        "operator_imp": args.operator_imp,
        "deg_scale": args.deg_scale,
        "noise_std": args.noise_std,
        "n_experts": args.n_experts,
        "n_reflections_per_expert": args.n_reflections_per_expert,
        "noise_opt_steps": args.noise_opt_steps,
        "lr_noise_opt": args.lr_noise_opt,
        "lambda_jb": args.lambda_jb,
        "lambda_ks": args.lambda_ks,
        "lambda_orth": args.lambda_orth,
        "phi": args.phi,
        "eta_tilde": args.eta_tilde,
        "use_amp": args.use_amp,
        "use_grad_checkpoint": args.use_grad_checkpoint,
        "workdir": str(args.workdir),
    })

    # ══════════════════════════════════════════════════════════════
    # Load solver with gradient checkpointing
    # ══════════════════════════════════════════════════════════════
    solver = get_solver(
        args.method, 
        dtype=torch.float32,
        use_gradient_checkpointing=args.use_grad_checkpoint
    )
    solver.seed = args.seed
    
    print(f"\n{'='*60}")
    print(f"  Memory Optimization Settings:")
    print(f"  - Gradient Checkpointing: {args.use_grad_checkpoint}")
    print(f"  - Mixed Precision (AMP): {args.use_amp}")
    print(f"  - Efficient Memory Mode: {args.efficient_memory}")
    print(f"  - Noise Parameterization: Direct z_init gradient update")
    print(f"  - Use Posterior Sampling: {args.use_posterior_sampling}")
    print(f"  - Optimization Sampler: EULER (not FireFlow)")
    print(f"{'='*60}\n")

    ###### suffix remove ############################
    def sanitize_prompt(s: str, suffix: str) -> str:
        s = s.strip()
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)].strip()
        return s

    # load text prompts
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file)

    if args.prompt_file is not None:
        prompts = [sanitize_prompt(p, args.prompt_suffix_to_remove) for p in prompts]

    solver.text_enc_1.to('cuda')
    solver.text_enc_2.to('cuda')
    solver.text_enc_3.to('cuda')

    if args.efficient_memory:
        with torch.no_grad():
            prompt_emb_set, pooled_emb_set = precompute(args, prompts, solver)
            null_emb, null_pooled_emb = solver.encode_prompt([''])

        del solver.text_enc_1
        del solver.text_enc_2
        del solver.text_enc_3
        torch.cuda.empty_cache()
        prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
        null_embs = [null_emb, null_pooled_emb]
    else:
        prompt_embs = [[None, None]] * len(prompts)
        null_embs = [None, None]

    print("Prompts are processed.")
    
    solver.vae.to('cuda')
    solver.transformer.to('cuda')

    #-----------------------problem setup------------------------
    device = solver.vae.device
    img_size = args.img_size

    if args.task == 'cs_walshhadamard':
        compress_by = round(1/args.deg_scale)
        from functions.svd_operators import WalshHadamardCS
        A_funcs = WalshHadamardCS(3, img_size, compress_by,
                                    torch.randperm(img_size ** 2, device=device), device)
    elif args.task == 'cs_blockbased':
        cs_ratio = args.deg_scale
        from functions.svd_operators import CS
        A_funcs = CS(3, img_size, cs_ratio, device)
    elif args.task == 'inpainting':
        from functions.svd_operators import Inpainting
        loaded = np.load("exp/inp_masks/FFHQ_mask.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(3, img_size, missing, device)
    elif args.task == 'inpainting_DIV2K':
        from functions.svd_operators import Inpainting
        loaded = np.load("exp/inp_masks/DIV2k_mask.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(3, img_size, missing, device)
    elif args.task == 'denoising':
        from functions.svd_operators import Denoising
        A_funcs = Denoising(3, img_size, device)
    elif args.task == 'colorization':
        from functions.svd_operators import Colorization
        A_funcs = Colorization(img_size, device)
    elif args.task == 'sr_averagepooling':
        blur_by = int(args.deg_scale)
        if args.operator_imp == 'SVD':
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(3, img_size, blur_by, device)
        else:
            raise NotImplementedError()
    elif args.task == 'sr_bicubic':
        factor = int(args.deg_scale)
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0
        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        
        if args.operator_imp == 'SVD':
            from functions.svd_operators import SRConv
            A_funcs = SRConv(kernel / kernel.sum(), 3, img_size, device, stride=factor)
        elif args.operator_imp == 'FFT':                
            from functions.fft_operators import Superres_fft, prepare_cubic_filter
            k = prepare_cubic_filter(1/factor)
            kernel = torch.from_numpy(k).float().to(device)
            A_funcs = Superres_fft(kernel / kernel.sum(), 3, img_size, device, stride=factor)
        else:
            raise NotImplementedError()
    elif args.task == 'deblur_uni':
        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(device), 3, img_size, device)
        elif args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
            A_funcs = Deblurring_fft(torch.Tensor([1 / 9] * 9).to(device), 3, img_size, device)
        else:
            raise NotImplementedError()
    elif args.task == 'deblur_gauss':
        sigma = args.deg_scale
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        size = 61
        ker = []
        for k in range(-size//2, size//2):
            ker.append(pdf(k))
        kernel = torch.Tensor(ker).to(device)

        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(kernel / kernel.sum(), 3, img_size, device)
        elif args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
            A_funcs = Deblurring_fft(kernel / kernel.sum(), 3, img_size, device)
        else:
            raise NotImplementedError()
    elif args.task == 'deblur_motion':
        from functions.motionblur.motionblur import Kernel
        if args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
        else:
            raise ValueError("set operator_imp = FFT")
    else:
        raise ValueError("degradation type not supported")

    # solve problem
    tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
    ])

    pbar = tqdm(get_img_list(args.img_path), desc="Solving")
    for i, path in enumerate(pbar):
        img_name = path.stem
        
        prompt_idx = i
        if prompt_idx >= len(prompt_embs):
            prompt_idx = len(prompt_embs) - 1

        img = tf(Image.open(path).convert('RGB'))
        img = img.unsqueeze(0).to(solver.vae.device)
        img = img * 2 - 1
        
        if args.task == 'deblur_motion':
            from functions.motionblur.motionblur import Kernel
            from functions.fft_operators import Deblurring_fft
            np.random.seed(seed=i * 10)
            kernel = torch.from_numpy(Kernel(size=(args.deg_scale, args.deg_scale), intensity=0.5).kernelMatrix)
            A_funcs = Deblurring_fft(kernel / kernel.sum(), 3, args.img_size, solver.transformer.device)
        
        y = A_funcs.A(img)
        y = y + args.noise_std * torch.randn(y.shape, device=y.device, generator=torch.Generator(y.device).manual_seed(args.seed))
        
        ################### MIXTURE OF HOUSEHOLDER EXPERTS ###################
        
        # Import functions
        from mixture_of_householder_experts import (
            MixtureOfHouseholderExperts,
            reg_jb, reg_ks,
            compute_gaussianity_tests
        )
        from noise_opt import compute_psnr_ssim, save_comparison
        from spatial_correlation import quick_spatial_check
        
        # ── 0. Ground-truth image preparation ────────────────────────
        gt_img = img.clone()
        gt_img_01 = (gt_img / 2 + 0.5).clamp(0, 1)

        # ── 1. Mixture of Householder Experts Parameterization ───────
        lH = args.img_size // solver.vae_scale_factor
        lW = args.img_size // solver.vae_scale_factor
        lC = solver.transformer.config.in_channels
        noise_shape = (1, lC, lH, lW)

        # LR Warmup scheduler
        def lr_lambda(step):
            # warmup_steps = 15
            # if step < warmup_steps:
            #     return 0.1 + 0.9 * (step / warmup_steps)
            return 1.0

        # scheduler = LambdaLR(optimizer_noise, lr_lambda)

        # ══════════════════════════════════════════════════════════════
        # Initialize GradScaler for Mixed Precision (if enabled)
        # ══════════════════════════════════════════════════════════════
        if args.use_amp:
            scaler = GradScaler(
                    init_scale=2.**7,        # 128 (기존 1024에서 낮춤)
                    growth_factor=1.5,       # 보수적 증가 (기존 2.0)
                    backoff_factor=0.5,
                    growth_interval=200,     # 더 천천히 키움 (기존 100)
                    enabled=True
                )
            print("✓ Mixed Precision (AMP) enabled")
            print("✓ Regularization losses computed in FP32 for numerical stability")
            print("✓ GradScaler initialized with init_scale=1024")

        # ── 2. Inner Euler timesteps (CHANGED FROM FIREFLOW) ──────────
        solver.scheduler.config.shift = 4.0
        solver.scheduler.set_timesteps(args.NFE + 1, device=solver.transformer.device)
        final_timesteps = solver.scheduler.timesteps
        inner_timesteps = final_timesteps[:args.inner_NFE+1]
        inner_timesteps[-1] = final_timesteps[-1]
        inner_sigmas = inner_timesteps.float() / solver.scheduler.config.num_train_timesteps

        # Prompt / null embeddings
        p_emb, p_pool = prompt_embs[prompt_idx]
        n_emb, n_pool = null_embs

        # ── 3. OPTIMIZATION LOOP WITH MOHE ────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Noise Optimization — Image [{img_name}]")
        print(f"  inner_NFE={args.inner_NFE}  opt_steps={args.noise_opt_steps}")
        print(f"  n_experts={args.n_experts}  n_reflections_per_expert={args.n_reflections_per_expert}")
        print(f"  lr={args.lr_noise_opt}")
        print(f"  Sampler: EULER (not FireFlow)")
        print(f"{'='*60}")

        sigma_for_dc = float(inner_sigmas[0])

        # ═══════════════════════════════════════════════════════════════
        # NEW: Save TWO types of progress images
        # ═══════════════════════════════════════════════════════════════
        z0t_progress_images = []          # Euler sampled images
        latent_noise_progress_images = []  # Direct latent noise decoded
        z_init = nn.Parameter(torch.randn(noise_shape, device=device))
        optimizer_noise = torch.optim.Adam([z_init], lr=args.lr_noise_opt)
        scheduler = LambdaLR(optimizer_noise, lr_lambda)

        for opt_iter in range(args.noise_opt_steps):
            optimizer_noise.zero_grad()

            # ══════════════════════════════════════════════════════════
            # MIXTURE OF EXPERTS OPTIMIZATION (USING EULER)
            # ══════════════════════════════════════════════════════════
            if args.use_amp:
                # (b) EULER and DC loss in mixed precision
                latent_noise = z_init

                with autocast(dtype=torch.float16):
                    z0t_hat = solver.euler_sample_wo_process_SDO(
                        initial_z=latent_noise,
                        timesteps=inner_timesteps, sigmas=inner_sigmas,
                        prompt_emb=p_emb, pooled_emb=p_pool,
                        null_emb=n_emb, null_pooled=n_pool,
                        cfg_scale=args.cfg_scale,
                    )

                    # z0t_hat_test1 = solver.euler_sample_wo_process(
                    #     initial_z=z0_test,
                    #     timesteps=inner_timesteps, sigmas=inner_sigmas,
                    #     prompt_emb=p_emb, pooled_emb=p_pool,
                    #     null_emb=n_emb, null_pooled=n_pool,
                    #     cfg_scale=args.cfg_scale,
                    # )

                    # z0t_hat_test2 = solver.euler_sample_wo_process(
                    #     initial_z=z0_test2,
                    #     timesteps=inner_timesteps, sigmas=inner_sigmas,
                    #     prompt_emb=p_emb, pooled_emb=p_pool,
                    #     null_emb=n_emb, null_pooled=n_pool,
                    #     cfg_scale=args.cfg_scale,
                    # )

                    loss_ir = solver.compute_data_consistency_loss(
                        z0t=z0t_hat, A=A_funcs, y=y,
                        sigma_val=sigma_for_dc, noise_std=args.noise_std,
                        phi=args.phi, eta_tilde=args.eta_tilde,
                        # save_debug_images=True,
                        # debug_save_path="./debug_freq_images"
                    )

                # (c) Regularization losses in FP32
                # MoHE regularization
                # loss_mohe_reg = hh_param.get_regularization_loss(
                #     lambda_orth=args.lambda_orth,
                #     lambda_scale=args.lambda_scale_reg,
                #     lambda_diversity=args.lambda_diversity
                # )
                
                # Gaussianity regularization
                loss_jb_v = reg_jb(latent_noise)
                loss_ks_v = reg_ks(latent_noise)

                loss_total = (loss_ir
                              + args.lambda_jb * loss_jb_v
                              + args.lambda_ks * loss_ks_v)

                # Check for inf/nan before backward
                if torch.isfinite(loss_total):
                    scaler.scale(loss_total).backward()
                    
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer_noise)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_([z_init], max_norm=1.0)
                    
                    # Store scale before step
                    scale_before = scaler.get_scale()
                    
                    scaler.step(optimizer_noise)
                    scaler.update()
                    scheduler.step()  # ← LR warmup

                    # with torch.no_grad():
                    #     cayley = LowRankCayleyLatent(latent_noise.data, rank=64).to(device)
                    #     latent_noise.data = cayley.get_x_T()

                    # Check if step was skipped
                    scale_after = scaler.get_scale()
                    if opt_iter < 10 and scale_before != scale_after:
                        print(f"  [opt {opt_iter}] GradScaler: scale {scale_before:.1f} → {scale_after:.1f}")
                else:
                    print(f"  ⚠ Warning: inf/nan detected in loss at iteration {opt_iter}, skipping update")
                    optimizer_noise.zero_grad()
            
            else:
                # Standard FP32 training
                latent_noise = z_init

                z0t_hat = solver.euler_sample_wo_process(
                    initial_z=latent_noise,
                    timesteps=inner_timesteps, sigmas=inner_sigmas,
                    prompt_emb=p_emb, pooled_emb=p_pool,
                    null_emb=n_emb, null_pooled=n_pool,
                    cfg_scale=args.cfg_scale,
                )

                loss_ir = solver.compute_data_consistency_loss_FAHG(
                    z0t=z0t_hat, A=A_funcs, y=y,
                    sigma_val=sigma_for_dc, noise_std=args.noise_std,
                    phi=args.phi, eta_tilde=args.eta_tilde,
                    # save_debug_images=True,
                    # debug_save_path="./debug_freq_images"
                )
                
                # Gaussianity regularization
                loss_jb_v = reg_jb(latent_noise)
                loss_ks_v = reg_ks(latent_noise)

                loss_total = (loss_ir
                              + args.lambda_jb * loss_jb_v
                              + args.lambda_ks * loss_ks_v)

                loss_total.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([z_init], max_norm=1.0)

                optimizer_noise.step()
                scheduler.step()  # ← LR warmup

            # ══════════════════════════════════════════════════════════
            # Save checkpoint images (TWO TYPES)
            # ══════════════════════════════════════════════════════════
            with torch.no_grad():
                # 1. Euler sampled image (z0t_hat)
                z0t_decoded = solver.decode(z0t_hat.detach()).float()
                z0t_decoded_01 = (z0t_decoded / 2 + 0.5).clamp(0, 1)
                z0t_progress_images.append(z0t_decoded_01.detach().cpu())
                
                # 2. Direct latent noise decoded (NEW!)
                latent_decoded = solver.decode(latent_noise.detach()).float()
                latent_decoded_01 = (latent_decoded / 2 + 0.5).clamp(0, 1)
                latent_noise_progress_images.append(latent_decoded_01.detach().cpu())
                
                # Clean up
                del z0t_decoded, z0t_decoded_01, latent_decoded, latent_decoded_01

            # ══════════════════════════════════════════════════════════
            # Logging and cleanup
            # ══════════════════════════════════════════════════════════
            if opt_iter % 1 == 0:
                with torch.no_grad():
                    gauss_tests = compute_gaussianity_tests(latent_noise.detach())
                    spatial_tests = quick_spatial_check(latent_noise.detach())
                    
                jb_tag = '✓ Gaussian' if gauss_tests['jb_gaussian'] else '✗ Non-Gaussian'
                ks_tag = '✓ Gaussian' if gauss_tests['ks_gaussian'] else '✗ Non-Gaussian'
                spatial_tag = '✓ Indep' if spatial_tests['independent'] else '✗ Corr'
                
                jb_pval = gauss_tests['jb_pvalue']
                ks_pval = gauss_tests['ks_pvalue']
                corr_max = spatial_tests['corr_max']

                # Memory monitoring
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3

                print(f"  [opt {opt_iter:4d}/{args.noise_opt_steps}]  "
                      f"total={loss_total.item():.4f}  ir={loss_ir.item():.4f}  "
                      f"jb={loss_jb_v.item():.2e}  ks={loss_ks_v.item():.2e}  "
                      f"JB:{jb_tag}(p={jb_pval:.4f})  KS:{ks_tag}(p={ks_pval:.4f})  "
                      f"Spatial:{spatial_tag}(corr={corr_max:.3f})  "
                      f"GPU:{allocated:.2f}/{reserved:.2f}GB")

                global_step = i * args.noise_opt_steps + opt_iter
                mlflow.log_metrics({
                    "loss_total": loss_total.item(),
                    "loss_ir": loss_ir.item(),
                    "loss_jb": loss_jb_v.item(),
                    "loss_ks": loss_ks_v.item(),
                }, step=global_step)

                torch.cuda.empty_cache()
                gc.collect()

            # Delete gradient tensors
            del z0t_hat, loss_ir, loss_jb_v, loss_ks_v, loss_total

        # ═══════════════════════════════════════════════════════════════
        # Save z0t progress grid (Euler sampled)
        # ═══════════════════════════════════════════════════════════════
        args.workdir.joinpath('progress_z0t').mkdir(parents=True, exist_ok=True)
        if len(z0t_progress_images) > 0:
            progress_tensors = torch.cat(z0t_progress_images, dim=0)
            n_total = progress_tensors.shape[0]
            nrow = min(n_total, max(1, int(math.ceil(math.sqrt(n_total)))))
            progress_grid = make_grid(progress_tensors, nrow=nrow, padding=2, pad_value=1.0)
            save_image(progress_grid, args.workdir / 'progress_z0t' / f'{img_name}_z0t_progress.png')
            print(f"  Saved z0t progress grid: {args.workdir / 'progress_z0t' / f'{img_name}_z0t_progress.png'}")
            del progress_tensors, progress_grid

        del z0t_progress_images
        
        # ═══════════════════════════════════════════════════════════════
        # NEW: Save latent_noise progress grid (Direct decode)
        # ═══════════════════════════════════════════════════════════════
        args.workdir.joinpath('progress_latent_noise').mkdir(parents=True, exist_ok=True)
        if len(latent_noise_progress_images) > 0:
            latent_progress_tensors = torch.cat(latent_noise_progress_images, dim=0)
            n_total = latent_progress_tensors.shape[0]
            nrow = min(n_total, max(1, int(math.ceil(math.sqrt(n_total)))))
            latent_progress_grid = make_grid(latent_progress_tensors, nrow=nrow, padding=2, pad_value=1.0)
            save_image(latent_progress_grid, args.workdir / 'progress_latent_noise' / f'{img_name}_latent_noise_progress.png')
            print(f"  Saved latent noise progress grid: {args.workdir / 'progress_latent_noise' / f'{img_name}_latent_noise_progress.png'}")
            del latent_progress_tensors, latent_progress_grid

        del latent_noise_progress_images
        
        torch.cuda.empty_cache()
        gc.collect()

        # ══════════════════════════════════════════════════════════════
        # Get optimized noise for final sampling
        # ══════════════════════════════════════════════════════════════
        with torch.no_grad():
            optimized_noise = z_init.detach()

        # ══════════════════════════════════════════════════════════════
        # 4-1. Final Euler Sampling (Original method)
        # ══════════════════════════════════════════════════════════════
        print(f"\n  Final Euler sampling (NFE={args.NFE}) ...")

        solver.scheduler.config.shift = 4.0
        solver.scheduler.set_timesteps(args.NFE + 1, device=solver.transformer.device)
        final_timesteps = solver.scheduler.timesteps
        final_sigmas = final_timesteps.float() / solver.scheduler.config.num_train_timesteps

        # euler_sample now returns (final_img, process_images)
        recon_img_euler, euler_process_images = solver.euler_sample(
            initial_z=optimized_noise,
            timesteps=final_timesteps, sigmas=final_sigmas,
            prompt_emb=p_emb, pooled_emb=p_pool,
            null_emb=n_emb, null_pooled=n_pool,
            cfg_scale=args.cfg_scale,
        )
        recon_img_euler_01 = (recon_img_euler / 2 + 0.5).clamp(0, 1)
        
        # ═══════════════════════════════════════════════════════════════
        # Save Euler sampling process grid
        # ═══════════════════════════════════════════════════════════════
        args.workdir.joinpath('progress_euler').mkdir(parents=True, exist_ok=True)
        if len(euler_process_images) > 0:
            # Concatenate all process images
            euler_progress_tensors = torch.cat(euler_process_images, dim=0)
            n_total = euler_progress_tensors.shape[0]
            nrow = min(n_total, max(1, int(math.ceil(math.sqrt(n_total)))))
            euler_progress_grid = make_grid(euler_progress_tensors, nrow=nrow, padding=2, pad_value=1.0)
            save_image(euler_progress_grid, args.workdir / 'progress_euler' / f'{img_name}_euler_process.png')
            print(f"  Saved Euler process grid ({n_total} steps): {args.workdir / 'progress_euler' / f'{img_name}_euler_process.png'}")
            del euler_progress_tensors, euler_progress_grid
        
        del euler_process_images
        torch.cuda.empty_cache()
        gc.collect()

        # ══════════════════════════════════════════════════════════════
        # 4-2. Posterior Sampling (if enabled)
        # ══════════════════════════════════════════════════════════════
        if args.use_posterior_sampling:
            print(f"\n  Posterior sampling (NFE={args.ps_NFE}) ...")
            
            recon_img_ps, ps_process_images = solver.posterior_sampling(
                measurement=y,
                operator=A_funcs,
                task=args.task,
                prompts=[prompts[prompt_idx]],
                NFE=args.ps_NFE,
                img_shape=(args.img_size, args.img_size),
                cfg_scale=args.cfg_scale,
                batch_size=1,
                latent=optimized_noise,  # Use optimized noise as initial point
                prompt_embs=[p_emb, p_pool],
                null_embs=[n_emb, n_pool],
                step_scale_ps_1=args.step_scale_ps_1,
                step_scale_ps_2=args.step_scale_ps_2,
                inner_steps=args.ps_inner_steps,
                sigma_y=args.noise_std,
                stochasticity_weight=args.stochasticity_weight,
                function_dc=args.function_dc,
            )
            
            recon_img_ps_01 = recon_img_ps
            
            # ═══════════════════════════════════════════════════════════════
            # Save Posterior sampling process grid
            # ═══════════════════════════════════════════════════════════════
            args.workdir.joinpath('progress_posterior').mkdir(parents=True, exist_ok=True)
            if len(ps_process_images) > 0:
                ps_progress_tensors = torch.cat(ps_process_images, dim=0)
                n_total = ps_progress_tensors.shape[0]
                nrow = min(n_total, max(1, int(math.ceil(math.sqrt(n_total)))))
                ps_progress_grid = make_grid(ps_progress_tensors, nrow=nrow, padding=2, pad_value=1.0)
                save_image(ps_progress_grid, args.workdir / 'progress_posterior' / f'{img_name}_posterior_process.png')
                print(f"  Saved Posterior process grid ({n_total} steps): {args.workdir / 'progress_posterior' / f'{img_name}_posterior_process.png'}")
                del ps_progress_tensors, ps_progress_grid
            
            del ps_process_images
            torch.cuda.empty_cache()
            gc.collect()
        else:
            recon_img_ps_01 = None
        
        # ── 5. Evaluate & Save ───────────────────────────────────────
        # Euler results
        metrics_euler = compute_psnr_ssim(recon_img_euler_01, gt_img_01)
        print(f"\n  ──── Euler Results [{img_name}] ────")
        print(f"  PSNR : {metrics_euler['psnr']:.2f} dB")
        print(f"  SSIM : {metrics_euler['ssim']:.4f}")
        print(f"  LPIPS: {metrics_euler['lpips']:.4f}")

        # Posterior sampling results (if enabled)
        if args.use_posterior_sampling:
            # FIXED: Move to GPU for metrics and save_comparison
            recon_img_ps_01 = recon_img_ps_01.to(device)

            metrics_ps = compute_psnr_ssim(recon_img_ps_01, gt_img_01)
            print(f"\n  ──── Posterior Sampling Results [{img_name}] ────")
            print(f"  PSNR : {metrics_ps['psnr']:.2f} dB")
            print(f"  SSIM : {metrics_ps['ssim']:.4f}")
            print(f"  LPIPS: {metrics_ps['lpips']:.4f}")

        if args.task in ['sr_bicubic', 'inpainting', 'inpainting_DIV2K', 'cs_walshhadamard', 'cs_blockbased']: # modify
            y_vis = A_funcs.At(y).reshape(1, 3, args.img_size, args.img_size)
        else:
            y_vis = y.reshape(1, 3, args.img_size, args.img_size)
        # y_vis = A_funcs.A_pinv(y).reshape(1, 3, args.img_size, args.img_size)
        y_vis_01 = (y_vis / 2 + 0.5).clamp(0, 1)

        # Save images
        save_image(gt_img_01, args.workdir / 'label' / f'{img_name}.png')
        save_image(recon_img_euler_01, args.workdir / 'recon_euler' / f'{img_name}.png')
        save_image(y_vis_01, args.workdir / 'input1' / f'{img_name}.png')
        
        if args.use_posterior_sampling:
            save_image(recon_img_ps_01, args.workdir / 'recon_posterior' / f'{img_name}.png')

        # Save comparisons
        comparison_path = args.workdir / 'recon_GT' / f'{img_name}_euler_comparison.png'
        save_comparison(
            y_vis_01, recon_img_euler_01, gt_img_01,
            comparison_path,
        )
        print(f"  Saved: {comparison_path}")

        if args.use_posterior_sampling:
            # FIXED: All tensors now on same device (GPU)
            ps_comparison_path = args.workdir / 'recon_GT' / f'{img_name}_posterior_comparison.png'
            save_comparison(
                y_vis_01, recon_img_ps_01, gt_img_01,
                ps_comparison_path,
            )
            print(f"  Saved: {ps_comparison_path}")

        # ── MLflow: log metrics and images ───────────────────────
        mlflow.log_metrics({
            f"{img_name}/psnr": metrics_euler['psnr'],
            f"{img_name}/ssim": metrics_euler['ssim'],
            f"{img_name}/lpips": metrics_euler['lpips'],
        }, step=i)
        if args.use_posterior_sampling:
            mlflow.log_metrics({
                f"{img_name}/psnr_ps": metrics_ps['psnr'],
                f"{img_name}/ssim_ps": metrics_ps['ssim'],
                f"{img_name}/lpips_ps": metrics_ps['lpips'],
            }, step=i)
        mlflow.log_artifact(str(comparison_path), artifact_path=f"images/{img_name}")
        mlflow.log_artifact(str(args.workdir / 'recon_euler' / f'{img_name}.png'), artifact_path=f"images/{img_name}")
        mlflow.log_artifact(str(args.workdir / 'input1' / f'{img_name}.png'), artifact_path=f"images/{img_name}")

        print(f"{'='*60}\n")

        # Clean up
        del z_init, optimizer_noise, optimized_noise
        del recon_img_euler, recon_img_euler_01
        if args.use_posterior_sampling:
            del recon_img_ps, recon_img_ps_01
        del gt_img, gt_img_01, y, y_vis, y_vis_01
        if args.use_amp:
            del scaler
        torch.cuda.empty_cache()
        gc.collect()

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--img_size', type=int, default=768)
    # workdir params
    parser.add_argument('--workdir', type=Path, default='workdir_mohe')
    parser.add_argument('--base_workdir', type=Path, default='workdir_mohe')
    # data params
    parser.add_argument('--img_path', type=Path)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=-1)
    # problem params
    parser.add_argument('--task', type=str, default='sr_bicubic')
    parser.add_argument('--method', type=str, default='naive')
    parser.add_argument('--deg_scale', type=int, default=4)
    parser.add_argument('--noise_std', type=float, default=0.03)
    # solver params
    parser.add_argument('--efficient_memory', default=False, action='store_true')
    parser.add_argument('--operator_imp', type=str, default="FFT", help="SVD | FFT")
    
    # ══════════════════════════════════════════════════════════════
    # MEMORY OPTIMIZATION PARAMS
    # ══════════════════════════════════════════════════════════════
    parser.add_argument('--use_grad_checkpoint', default=False, action='store_true',
                        help='Enable gradient checkpointing (saves 30-50%% memory, -20%% speed)')
    parser.add_argument('--use_amp', default=False, action='store_true',
                        help='Enable mixed precision training (saves 20-30%% memory, +30%% speed)')
    
    # ══════════════════════════════════════════════════════════════
    # MIXTURE OF HOUSEHOLDER EXPERTS PARAMS
    # ══════════════════════════════════════════════════════════════
    parser.add_argument('--n_experts', type=int, default=8,
                        help='Number of independent experts (default: 8)')
    parser.add_argument('--n_reflections_per_expert', type=int, default=4,
                        help='Number of Householder reflections per expert (default: 4)')
    
    # Noise Optimization params
    parser.add_argument('--inner_NFE', type=int, default=5,
                        help='NFE for Euler sampling inside noise opt loop')
    parser.add_argument('--noise_opt_steps', type=int, default=50,
                        help='Number of noise optimization iterations (reduced from 100 with MoHE)')
    parser.add_argument('--lr_noise_opt', type=float, default=1e-2,
                        help='Learning rate for optimization')
    
    # MoHE Regularization params
    parser.add_argument('--lambda_orth', type=float, default=1e-4,
                        help='Weight for orthogonality regularization (per expert)')
    parser.add_argument('--lambda_scale_reg', type=float, default=1e-5,
                        help='Weight for diagonal scale regularization')
    parser.add_argument('--lambda_diversity', type=float, default=1e-3,
                        help='Weight for expert diversity regularization')
    
    # Gaussianity Regularization params
    parser.add_argument('--lambda_jb', type=float, default=1e-2,
                        help='Weight for Jarque-Bera regularization')
    parser.add_argument('--lambda_ks', type=float, default=1e-2,
                        help='Weight for KS quantile-matching regularization')
    
    # Data Consistency params
    parser.add_argument('--phi', type=float, default=1.0,
                        help='Exponent for data consistency weighting schedule')
    parser.add_argument('--eta_tilde', type=float, default=0.8,
                        help='Regularization coefficient for pseudo-inverse in DC loss')
    
    # ══════════════════════════════════════════════════════════════
    # POSTERIOR SAMPLING PARAMS (NEW!)
    # ══════════════════════════════════════════════════════════════
    parser.add_argument('--use_posterior_sampling', default=False, action='store_true',
                        help='Enable posterior sampling after optimization')
    parser.add_argument('--ps_NFE', type=int, default=50,
                        help='NFE for posterior sampling')
    parser.add_argument('--step_scale_ps_1', type=float, default=1.0,
                        help='Starting step scale for posterior sampling')
    parser.add_argument('--step_scale_ps_2', type=float, default=1.0,
                        help='Ending step scale for posterior sampling')
    parser.add_argument('--ps_inner_steps', type=int, default=1,
                        help='Inner steps for data consistency in posterior sampling')
    parser.add_argument('--stochasticity_weight', type=float, default=0.5,
                        help='Stochasticity weight for posterior sampling')
    parser.add_argument('--function_dc', type=str, default='linear',
                        choices=['linear', 'exponential', 'logarithm', 'constant'],
                        help='Function for data consistency schedule in posterior sampling')
    
    parser.add_argument('--prompt_suffix_to_remove', type=str, default=', high-resolution, 8k')

    args = parser.parse_args()

    # workdir creation and seed setup
    set_seed(args.seed)
    args.workdir.joinpath('input1').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('recon_euler').mkdir(parents=True, exist_ok=True)
    if args.use_posterior_sampling:
        args.workdir.joinpath('recon_posterior').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('recon_GT').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('label').mkdir(parents=True, exist_ok=True)

    # run main script
    run(args)