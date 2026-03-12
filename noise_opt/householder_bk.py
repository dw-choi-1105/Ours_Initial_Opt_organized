"""
Householder Noise Parameterization & Gaussianity Regularization

Parameterizes initial noise z as a sequence of Householder reflections
applied to a fixed Gaussian vector. This preserves the norm (and thus
Gaussianity) while allowing gradient-based optimization of the noise.

Adapted from BIRD (Blind Image Restoration via fast Diffusion inversion).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════════
# Householder Noise Parameterization
# ══════════════════════════════════════════════════════════════

class HouseholderNoiseParam(nn.Module):
    """
    z_optimized = H_K · H_{K-1} · ... · H_1 · z_fixed

    Each H_i(x) = x - 2·<x,v_i>/<v_i,v_i> · v_i  is a Householder
    reflection.  Only the direction vectors {v_i} are learnable.
    Because reflections are orthogonal transforms, ||z_optimized|| = ||z_fixed||
    and the marginal distribution stays N(0, I).
    """

    def __init__(self, noise_shape: tuple, n_reflections: int = 32):
        super().__init__()
        self.noise_shape   = noise_shape
        self.n_reflections = n_reflections
        self.dim           = int(np.prod(noise_shape))

        z = torch.randn(*noise_shape)
        self.register_buffer('z_fixed', z)
        self.vs = nn.Parameter(torch.randn(n_reflections, self.dim))

    def apply_single_householder(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_norm_sq = (v * v).sum() + 1e-9
        coeff     = 2.0 * (x * v).sum() / v_norm_sq
        return x - coeff * v

    def forward(self) -> torch.Tensor:
        x = self.z_fixed.flatten()
        for i in range(self.n_reflections):
            x = self.apply_single_householder(x, self.vs[i])
        return x.reshape(self.noise_shape)


# ══════════════════════════════════════════════════════════════
# Differentiable Gaussianity Regularization Losses
# ══════════════════════════════════════════════════════════════

def reg_orthogonality(vs: torch.Tensor) -> torch.Tensor:
    """
    Encourage Householder vectors to be orthonormal: ||V V^T - I||_F^2 / d^2.
    When vs are orthonormal, each reflection touches an independent direction,
    preventing correlated distortion of the latent.
    """
    K, d = vs.shape
    VVt = vs @ vs.t()
    I_K = torch.eye(K, device=vs.device, dtype=vs.dtype)
    return ((VVt - I_K) ** 2).mean() / (d ** 2)


def reg_jb(latent: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Jarque-Bera surrogate: S^2 + K^2/4.
    S = skewness, K = excess kurtosis.  Both vanish for N(0,1).
    """
    z     = latent.flatten().float()
    mu    = z.mean()
    sigma = z.std() + 1e-8
    z_std = (z - mu) / sigma
    S = (z_std ** 3).mean()           # skewness
    K = (z_std ** 4).mean() - 3.0     # excess kurtosis
    return S ** 2 + (K ** 2) / 4.0


# --- KS quantile-matching helpers ---

_theory_quantiles_cache: dict = {}

def _get_theory_quantiles(n_quantiles: int, device: torch.device) -> torch.Tensor:
    key = (n_quantiles, str(device))
    if key not in _theory_quantiles_cache:
        probs = np.linspace(0.5 / n_quantiles, 1.0 - 0.5 / n_quantiles, n_quantiles)
        _theory_quantiles_cache[key] = torch.tensor(
            scipy_stats.norm.ppf(probs), dtype=torch.float32, device=device
        )
    return _theory_quantiles_cache[key]


def reg_ks(latent: torch.Tensor, n_quantiles: int = 50) -> torch.Tensor:
    """
    Differentiable KS-like loss: L2 distance between empirical quantiles
    and theoretical N(0,1) quantiles.  torch.sort is differentiable,
    so gradients flow back to the Householder parameters.
    """
    z           = latent.flatten().float()
    n           = z.shape[0]
    z_sorted, _ = torch.sort(z)
    indices     = torch.linspace(0, n - 1, n_quantiles, device=z.device).long()
    empirical_q = z_sorted[indices]
    theory_q    = _get_theory_quantiles(n_quantiles, z.device)
    return ((empirical_q - theory_q) ** 2).mean()


# ══════════════════════════════════════════════════════════════
# Gaussianity Statistical Tests (non-differentiable, for logging)
# ══════════════════════════════════════════════════════════════

def compute_gaussianity_tests(latent: torch.Tensor) -> dict:
    """
    Run JB and KS tests on the latent to check Gaussianity.
    Returns dict with p-values and boolean pass/fail (threshold=0.05).
    """
    z = latent.detach().cpu().float().flatten().numpy()
    jb_stat, jb_pvalue = scipy_stats.jarque_bera(z)
    ks_stat, ks_pvalue = scipy_stats.kstest(z, 'norm', args=(0, 1))
    return {
        'jb_pvalue':  jb_pvalue,
        'jb_gaussian': jb_pvalue > 0.05,
        'ks_pvalue':  ks_pvalue,
        'ks_gaussian': ks_pvalue > 0.05,
    }