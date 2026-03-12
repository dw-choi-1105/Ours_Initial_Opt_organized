"""
noise_opt — Initial Noise Optimization for Rectified Flow Models

Modules:
    householder : Householder noise parameterization & Gaussianity regularization
    metrics     : PSNR / SSIM evaluation & image saving utilities

Note: Sampling functions (fireflow_sample, euler_sample) and
      data-consistency loss (compute_data_consistency_loss) are
      methods of SD3Euler in sd3_sampler.py, accessible via `solver.*`.
"""

# from .householder import (
#     HouseholderNoiseParam,
#     reg_orthogonality,
#     reg_jb,
#     reg_ks,
#     compute_gaussianity_tests,
# )
from .metrics import compute_psnr_ssim, save_comparison