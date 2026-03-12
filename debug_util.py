from pathlib import Path
import torch
from torchvision.utils import save_image

# ──────────────────────────────────────────────────────────────
# Debug image saver
# Usage:
#   debug_save(tensor, "step_name")                 # auto range detection
#   debug_save(tensor, "step_name", range="01")     # [0, 1] tensor
#   debug_save(tensor, "step_name", range="11")     # [-1, 1] tensor (img/recon)
#   debug_save(tensor, "step_name", range="latent") # latent (normalize per-channel)
# ──────────────────────────────────────────────────────────────
_DEBUG_DIR = Path("debug_img")

def debug_save(tensor: torch.Tensor, name: str, range: str = "auto"):
    _DEBUG_DIR.mkdir(exist_ok=True)
    t = tensor.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]  # (C, H, W)

    if range == "auto":
        vmin, vmax = t.min().item(), t.max().item()
        if vmin >= -0.1 and vmax <= 1.1:
            range = "01"
        elif vmin >= -1.1 and vmax <= 1.1:
            range = "11"
        else:
            range = "latent"

    if range == "11":
        t = (t / 2 + 0.5).clamp(0, 1)
    elif range == "latent":
        # normalize each channel independently to [0, 1]
        t_min = t.flatten(1).min(dim=1).values[:, None, None]
        t_max = t.flatten(1).max(dim=1).values[:, None, None]
        t = ((t - t_min) / (t_max - t_min + 1e-8)).clamp(0, 1)
        if t.shape[0] > 3:
            t = t[:3]  # show first 3 channels as RGB
    # "01" needs no conversion

    save_path = _DEBUG_DIR / f"{name}.png"
    save_image(t, save_path)
    print(f"  [debug] saved → {save_path}  shape={list(tensor.shape)}  range=[{tensor.min():.3f}, {tensor.max():.3f}]")
