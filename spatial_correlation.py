"""
Spatial Correlation Tests for Gaussian Noise

Measures spatial independence of latent noise to detect over-smoothing
"""

import numpy as np
import torch
from scipy.stats import pearsonr


def compute_spatial_autocorrelation(latent: torch.Tensor) -> dict:
    """
    Compute spatial autocorrelation at lag 1.
    
    Independent noise should have near-zero autocorrelation.
    High autocorrelation indicates spatial smoothing (bad for reconstruction).
    
    Args:
        latent: (B, C, H, W) tensor
    
    Returns:
        dict with autocorrelation metrics
    """
    # Use first channel for analysis
    z = latent[0, 0, :, :].detach().cpu().numpy()
    
    # Horizontal autocorrelation (lag 1)
    z_left = z[:, :-1].flatten()
    z_right = z[:, 1:].flatten()
    corr_h, _ = pearsonr(z_left, z_right)
    
    # Vertical autocorrelation (lag 1)
    z_top = z[:-1, :].flatten()
    z_bottom = z[1:, :].flatten()
    corr_v, _ = pearsonr(z_top, z_bottom)
    
    # Maximum (most conservative)
    max_corr = max(abs(corr_h), abs(corr_v))
    
    return {
        'horizontal': corr_h,
        'vertical': corr_v,
        'max': max_corr,
        'independent': max_corr < 0.1  # Threshold for independence
    }


def compute_morans_i(latent: torch.Tensor) -> float:
    """
    Compute Moran's I statistic for spatial autocorrelation.
    
    Moran's I ∈ [-1, 1]:
    - I ≈ 0: No spatial correlation (good)
    - I > 0: Positive correlation (similar values cluster - bad)
    - I < 0: Negative correlation (rare)
    
    Args:
        latent: (B, C, H, W) tensor
    
    Returns:
        Moran's I statistic
    """
    z = latent[0, 0, :, :].detach().cpu().numpy()
    H, W = z.shape
    n = H * W
    
    z_flat = z.flatten()
    z_mean = z_flat.mean()
    
    # Numerator: weighted cross-products with 4-neighbors
    numerator = 0
    weight_sum = 0
    
    for i in range(H):
        for j in range(W):
            z_i = z[i, j]
            
            # 4-connected neighbors
            neighbors = []
            if i > 0:
                neighbors.append(z[i-1, j])
            if i < H-1:
                neighbors.append(z[i+1, j])
            if j > 0:
                neighbors.append(z[i, j-1])
            if j < W-1:
                neighbors.append(z[i, j+1])
            
            for z_j in neighbors:
                numerator += (z_i - z_mean) * (z_j - z_mean)
                weight_sum += 1
    
    # Denominator
    denominator = np.sum((z_flat - z_mean) ** 2)
    
    # Moran's I
    morans_i = (n / weight_sum) * (numerator / denominator)
    
    return morans_i


def compute_psd_flatness(latent: torch.Tensor) -> float:
    """
    Compute Power Spectral Density flatness.
    
    White noise has flat PSD (all frequencies equally represented).
    Correlated noise has non-flat PSD (some frequencies dominate).
    
    Args:
        latent: (B, C, H, W) tensor
    
    Returns:
        Flatness measure (0-1, higher is better)
    """
    z = latent[0, 0, :, :].detach().cpu().numpy()
    
    # 2D FFT
    fft = np.fft.fft2(z)
    psd = np.abs(fft) ** 2
    
    # Flatness: geometric mean / arithmetic mean
    # For white noise, this is close to 1
    psd_positive = psd[psd > 0]
    
    geometric_mean = np.exp(np.mean(np.log(psd_positive + 1e-10)))
    arithmetic_mean = np.mean(psd_positive)
    
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    
    return flatness


def comprehensive_spatial_test(latent: torch.Tensor, verbose: bool = False) -> dict:
    """
    Comprehensive spatial independence test.
    
    Combines multiple methods for robust assessment.
    
    Args:
        latent: (B, C, H, W) tensor
        verbose: If True, print detailed results
    
    Returns:
        dict with all test results and overall assessment
    """
    # 1. Autocorrelation
    autocorr = compute_spatial_autocorrelation(latent)
    
    # 2. Moran's I
    morans_i = compute_morans_i(latent)
    
    # 3. PSD Flatness
    psd_flat = compute_psd_flatness(latent)
    
    # Overall assessment
    results = {
        'autocorr_h': autocorr['horizontal'],
        'autocorr_v': autocorr['vertical'],
        'autocorr_max': autocorr['max'],
        'morans_i': morans_i,
        'psd_flatness': psd_flat,
        'is_independent': (
            autocorr['max'] < 0.1 and 
            abs(morans_i) < 0.1 and 
            psd_flat > 0.6
        )
    }
    
    if verbose:
        print(f"  Spatial Tests:")
        print(f"    Autocorr (H/V): {autocorr['horizontal']:.3f} / {autocorr['vertical']:.3f}")
        print(f"    Moran's I: {morans_i:.3f}")
        print(f"    PSD Flatness: {psd_flat:.3f}")
        print(f"    Independent: {results['is_independent']}")
    
    return results


def quick_spatial_check(latent: torch.Tensor) -> dict:
    """
    Quick spatial independence check (optimized for speed).
    
    Only computes autocorrelation (fastest method).
    
    Args:
        latent: (B, C, H, W) tensor
    
    Returns:
        dict with autocorrelation and independence flag
    """
    autocorr = compute_spatial_autocorrelation(latent)
    
    return {
        'corr_h': autocorr['horizontal'],
        'corr_v': autocorr['vertical'],
        'corr_max': autocorr['max'],
        'independent': autocorr['independent']
    }