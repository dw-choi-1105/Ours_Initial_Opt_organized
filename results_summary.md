# Experiment Results Summary

## deblur_motion — FFHQ_5 (2026-03-14)

| Experiment | Script | inner_NFE | opt_steps | λ_tc_cons | PSNR (dB) | SSIM | GPU Memory |
|---|---|---|---|---|---|---|---|
| euler | solve_ours_noise_opt_euler.py | 5 | 50 | — | 20.49 | 0.7606 | 30.44 GB |
| euler_sdo | solve_ours_noise_opt_euler_sdo.py | 5 | 50 | — | 18.26 | 0.7094 | 30.23 GB |
| **euler+TC λ=1** | solve_ours_noise_opt_euler_consistency_regularization.py | 5 | 50 | 1.0 | 21.59 | 0.7516 | 30.44 GB |
| **euler+TC λ=10** | solve_ours_noise_opt_euler_consistency_regularization.py | 5 | 50 | 10.0 | 21.94 | 0.7585 | 30.44 GB |
| **euler+TC λ=100** | solve_ours_noise_opt_euler_consistency_regularization.py | 5 | 50 | 100.0 | **25.03** | **0.7864** | 30.44 GB |
| **euler+TC λ=1000** | solve_ours_noise_opt_euler_consistency_regularization.py | 5 | 50 | 1000.0 | 22.06 | 0.6876 | 30.44 GB |
| **sdo+TC λ=100** | solve_ours_noise_opt_euler_sdo_consistency_regularization.py | 5 | 50 | 100.0 | 22.05 | 0.7503 | 30.44 GB |

### Per-image PSNR (dB) — euler+TC grid search

| Image | euler (λ=0) | TC λ=1 | TC λ=10 | TC λ=100 | TC λ=1000 |
|---|---|---|---|---|---|
| 00000 | — | 24.67 | 24.68 | **26.27** | 14.55 |
| 00004 | — | 18.05 | 18.87 | **21.95** | 22.17 |
| 00010 | — | 19.93 | 20.09 | **24.49** | 26.54 |
| 00015 | — | 23.31 | 23.61 | **25.74** | 24.32 |
| 00023 | — | 21.99 | 22.43 | **26.70** | 22.71 |
| **avg** | 20.49 | 21.59 | 21.94 | **25.03** | 22.06 |

### Per-image PSNR (dB) — sdo+TC λ=100

| Image | PSNR (dB) | SSIM |
|---|---|---|
| 00000 | 22.73 | 0.7469 |
| 00004 | 21.43 | 0.6900 |
| 00010 | 22.19 | 0.7504 |
| 00015 | 23.32 | 0.8074 |
| 00023 | 20.57 | 0.7566 |
| **avg** | **22.05** | **0.7503** |

### Key Findings

- **Best configuration: euler+TC λ=100** (+4.54 dB over euler baseline)
- λ=1000 shows over-regularization (image 00000: 14.55 dB)
- SDO+TC λ=100 improves over SDO baseline but underperforms Euler+TC
  - SDO only backpropagates through step 0 → TC gradient signal is weaker

### Common Settings
- img_path: `./data/FFHQ_5` (5 images)
- img_size: 768, NFE: 28, seed: 42
- task: deblur_motion, operator_imp: FFT, deg_scale: 61, noise_std: 0.03
- n_experts: 8, n_reflections_per_expert: 4
- lr_noise_opt: 0.001, lambda_jb: 0, lambda_ks: 0, lambda_orth: 0
- --use_grad_checkpoint, --use_amp, --efficient_memory
- TC params: tc_num_samples=2, tc_t_min=0.1, tc_t_max=0.9
