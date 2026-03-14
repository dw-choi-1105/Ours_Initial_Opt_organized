"""
Evaluation Metrics & Image Saving Utilities

- tensor_to_uint8:   Convert [-1,1] or [0,1] image tensor to uint8 numpy
- compute_psnr_ssim: Compute PSNR, SSIM, and LPIPS between two [0,1] tensors
- save_comparison:   Save side-by-side [measurement | recon | GT] grid
"""

import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

# Lazy LPIPS model cache (keyed by device string to avoid reloading)
_lpips_cache: dict = {}


def _get_lpips_fn(device: torch.device):
    key = str(device)
    if key not in _lpips_cache:
        import lpips
        _lpips_cache[key] = lpips.LPIPS(net='alex').to(device)
    return _lpips_cache[key]


def tensor01_to_uint8(img_01: torch.Tensor) -> np.ndarray:
    """
    Convert a [0, 1] image tensor of shape (1, C, H, W) or (C, H, W)
    to a uint8 numpy array of shape (H, W, C).
    """
    if img_01.dim() == 4:
        img_01 = img_01[0]
    return (img_01.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def compute_psnr_ssim(
    recon_01: torch.Tensor,
    gt_01: torch.Tensor,
) -> dict:
    """
    Compute PSNR (dB), SSIM, and LPIPS between two [0, 1]-range image tensors.

    Returns:
        {'psnr': float, 'ssim': float, 'lpips': float}
    """
    gt_np  = tensor01_to_uint8(gt_01)
    rec_np = tensor01_to_uint8(recon_01)

    psnr_val = float(_psnr(gt_np, rec_np, data_range=255))
    ssim_val = float(_ssim(gt_np, rec_np, data_range=255, channel_axis=-1))

    # LPIPS: expects (N, C, H, W) in [-1, 1]
    r = recon_01 * 2.0 - 1.0
    g = gt_01 * 2.0 - 1.0
    if r.dim() == 3:
        r, g = r.unsqueeze(0), g.unsqueeze(0)
    lpips_fn = _get_lpips_fn(recon_01.device)
    with torch.no_grad():
        lpips_val = float(lpips_fn(r, g).item())

    return {'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}


def save_comparison(
    y_vis_01: torch.Tensor,
    recon_01: torch.Tensor,
    gt_01: torch.Tensor,
    save_path,
    nrow: int = 3,
):
    """
    Save a side-by-side grid image: [measurement | reconstruction | GT].
    All inputs should be [0, 1]-range tensors of shape (1, C, H, W).
    """
    grid = make_grid(
        torch.cat([y_vis_01, recon_01, gt_01], dim=0),
        nrow=nrow, padding=4, pad_value=1.0,
    )
    save_image(grid, save_path)
