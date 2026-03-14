# Experiment Results Summary

## deblur_motion — FFHQ_5 (2026-03-14)

| Experiment | Script | inner_NFE | noise_opt_steps | PSNR (dB) | SSIM | Time | GPU Memory |
|---|---|---|---|---|---|---|---|
| euler | solve_ours_noise_opt_euler.py | 5 | 50 | **20.49** | **0.7606** | 8m 02s | 30.44 GB |
| euler_sdo | solve_ours_noise_opt_euler_sdo.py | 5 | 50 | 18.26 | 0.7094 | 7m 17s | 30.23 GB |

### Common Settings
- img_path: `./data/FFHQ_5` (5 images)
- img_size: 768
- NFE: 28, seed: 42
- task: deblur_motion, operator_imp: FFT, deg_scale: 61, noise_std: 0.03
- n_experts: 8, n_reflections_per_expert: 4
- lr_noise_opt: 0.001, lambda_jb: 0, lambda_ks: 0, lambda_orth: 0
- --use_grad_checkpoint, --use_amp, --efficient_memory
