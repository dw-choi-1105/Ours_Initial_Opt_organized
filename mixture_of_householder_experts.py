"""
Mixture of Householder Experts (MoHE)
Each basis has its own independent Householder + Scaling transformation

z = Σ w_i · [D_i · (H_K_i · ... · H_1_i · basis_i)]

This is essentially a "Mixture of Experts" approach where each expert
is a Householder + Scaling path operating on a fixed Gaussian basis.

Advantages:
- Multiple independent search directions
- Automatic selection via learned weights
- Robust to local minima
- Graceful degradation (bad experts get weight → 0)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from typing import Optional


class MixtureOfHouseholderExperts(nn.Module):
    """
    Mixture of Householder Experts for noise parameterization.
    
    Each expert transforms a fixed Gaussian basis independently:
        expert_i = D_i · (H_K_i · ... · H_1_i · basis_i)
    
    Final noise is weighted combination:
        z = Σ w_i · expert_i
    
    Parameters:
        noise_shape: Shape of noise tensor (e.g., (1, 16, 96, 96))
        n_experts: Number of independent experts (default: 8)
                   Recommended: 4-16 for good diversity vs. efficiency
        n_reflections_per_expert: Householder reflections per expert (default: 4)
                                  Smaller than standard (32) since we have multiple experts
        use_scaling: Whether to use diagonal scaling D_i (default: True)
    
    Total parameters:
        - Weights: n_experts
        - Householder vectors: n_experts × n_reflections × dim
        - Scales (if enabled): n_experts × dim
    
    Example:
        # 8 experts, 4 reflections each = 32 total transformations
        # But with 8 independent paths!
        hh_param = MixtureOfHouseholderExperts(
            noise_shape=(1, 16, 96, 96),
            n_experts=8,
            n_reflections_per_expert=4
        )
    """
    
    def __init__(
        self,
        noise_shape: tuple,
        n_experts: int = 8,
        n_reflections_per_expert: int = 4,
        use_scaling: bool = True
    ):
        super().__init__()
        self.noise_shape = noise_shape
        self.n_experts = n_experts
        self.n_reflections = n_reflections_per_expert
        self.use_scaling = use_scaling
        self.dim = int(np.prod(noise_shape))
        
        # ── Fixed Gaussian bases (one per expert) ──
        bases = torch.randn(n_experts, *noise_shape)
        # Normalize each basis
        for i in range(n_experts):
            bases[i] = bases[i] / bases[i].flatten().norm()
        self.register_buffer('bases', bases)
        
        # ── Expert-specific Householder vectors ──
        # Shape: (n_experts, n_reflections_per_expert, dim)
        self.expert_vs = nn.Parameter(
            torch.randn(n_experts, n_reflections_per_expert, self.dim) * 0.05
        )
        
        # ── Expert-specific diagonal scaling (optional) ──
        if use_scaling:
            # Initialize as identity (log(1) = 0)
            self.expert_log_scales = nn.Parameter(
                torch.zeros(n_experts, self.dim)
            )
        
        # ── Expert mixing weights ──
        # Initialize with slight preference for first expert (warm start)
        init_logits = torch.randn(n_experts) * 0.1
        # init_logits[0] = 1.0
        self.expert_logits = nn.Parameter(init_logits)
    
    def apply_single_householder(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Apply single Householder reflection"""
        v_norm_sq = (v * v).sum() + 1e-9
        coeff = 2.0 * (x * v).sum() / v_norm_sq
        return x - coeff * v
    
    def transform_expert(self, expert_id: int) -> torch.Tensor:
        """
        Transform the basis for a single expert.
        
        Returns: expert_i = D_i · (H_K_i · ... · H_1_i · basis_i)
        """
        # Start with fixed basis
        x = self.bases[expert_id].flatten()
        
        # Apply expert's Householder reflections
        for refl_id in range(self.n_reflections):
            v = self.expert_vs[expert_id, refl_id]
            x = self.apply_single_householder(x, v)
        
        # Apply expert's diagonal scaling (if enabled)
        if self.use_scaling:
            scales = torch.exp(self.expert_log_scales[expert_id].clamp(-0.5, 0.5))
            scales = scales / scales.mean()  # Preserve variance
            x = x * scales
        
        # Normalize to preserve Gaussianity
        x = x / x.std() * np.sqrt(self.dim / (self.dim - 1))
        
        return x
    
    def forward(self) -> torch.Tensor:
        """
        Compute final noise as weighted mixture of experts.
        
        z = Σ w_i · expert_i
        """
        # Compute mixing weights (normalized)
        weights = torch.softmax(self.expert_logits, dim=0)
        weights = weights / weights.norm()  # L2 normalization
        
        # Compute each expert's output
        experts_output = []
        for i in range(self.n_experts):
            expert_i = self.transform_expert(i)
            experts_output.append(expert_i)
        
        # Weighted combination
        experts_tensor = torch.stack(experts_output, dim=0)  # (n_experts, dim)
        z = torch.sum(weights[:, None] * experts_tensor, dim=0)
        
        # Final normalization
        target_norm = np.sqrt(self.dim)
        z = z * (target_norm / (z.norm() + 1e-9))
        
        return z.reshape(self.noise_shape)
    
    def get_expert_importance(self) -> dict:
        """
        Analyze which experts are most important.
        Returns dict with expert indices and their normalized weights.
        """
        weights = torch.softmax(self.expert_logits, dim=0)
        weights_normalized = weights / weights.norm()
        
        # Sort by importance
        sorted_weights, sorted_indices = torch.sort(
            weights_normalized, descending=True
        )
        
        return {
            'weights': weights_normalized.detach().cpu().numpy(),
            'top_experts': [
                (idx.item(), weight.item()) 
                for idx, weight in zip(sorted_indices, sorted_weights)
            ]
        }
    
    def get_regularization_loss(
        self,
        lambda_orth: float = 1e-4,
        lambda_scale: float = 1e-5,
        lambda_diversity: float = 1e-3
    ) -> torch.Tensor:
        """
        Regularization losses for the mixture model.
        
        Args:
            lambda_orth: Orthogonality for Householder vectors (per expert)
            lambda_scale: Prevent scales from deviating too much from 1
            lambda_diversity: Encourage experts to be different
        
        Returns:
            Total regularization loss
        """
        total_loss = 0.0
        
        # 1. Orthogonality within each expert
        if lambda_orth > 0:
            for i in range(self.n_experts):
                vs = self.expert_vs[i]  # (n_reflections, dim)
                K, d = vs.shape
                VVt = vs @ vs.t()
                I_K = torch.eye(K, device=vs.device, dtype=vs.dtype)
                loss_orth_i = ((VVt - I_K) ** 2).mean() / (d ** 2)
                total_loss = total_loss + lambda_orth * loss_orth_i
        
        # 2. Scale regularization
        if self.use_scaling and lambda_scale > 0:
            # Encourage scales to stay near 1 (log(1) = 0)
            loss_scale = (self.expert_log_scales ** 2).mean()
            total_loss = total_loss + lambda_scale * loss_scale
        
        # 3. Diversity regularization (encourage different experts)
        if lambda_diversity > 0:
            # Compute pairwise similarity between expert outputs
            with torch.no_grad():
                expert_outputs = []
                for i in range(self.n_experts):
                    expert_outputs.append(self.transform_expert(i))
                expert_matrix = torch.stack(expert_outputs, dim=0)  # (n_experts, dim)
            
            # Correlation matrix
            expert_matrix_normalized = expert_matrix / expert_matrix.norm(dim=1, keepdim=True)
            correlation = expert_matrix_normalized @ expert_matrix_normalized.t()
            
            # Penalize high off-diagonal correlations
            mask = ~torch.eye(self.n_experts, dtype=torch.bool, device=correlation.device)
            off_diagonal_corr = correlation[mask].abs().mean()
            
            total_loss = total_loss + lambda_diversity * off_diagonal_corr
        
        return total_loss


# ══════════════════════════════════════════════════════════════
# Lightweight version for faster optimization
# ══════════════════════════════════════════════════════════════

class LightMixtureOfExperts(nn.Module):
    """
    Lightweight version: No Householder, just scaling per basis
    
    z = Σ w_i · (D_i · basis_i)
    
    Much faster than full MoHE, still provides diversity benefits.
    Recommended for initial experiments.
    """
    
    def __init__(
        self,
        noise_shape: tuple,
        n_experts: int = 16
    ):
        super().__init__()
        self.noise_shape = noise_shape
        self.n_experts = n_experts
        self.dim = int(np.prod(noise_shape))
        
        # Fixed bases
        bases = torch.randn(n_experts, *noise_shape)
        for i in range(n_experts):
            bases[i] = bases[i] / bases[i].flatten().norm()
        self.register_buffer('bases', bases)
        
        # Expert-specific diagonal scaling
        self.expert_log_scales = nn.Parameter(torch.zeros(n_experts, self.dim))
        
        # Mixing weights
        init_logits = torch.randn(n_experts) * 0.1
        init_logits[0] = 1.0
        self.expert_logits = nn.Parameter(init_logits)
    
    def forward(self) -> torch.Tensor:
        weights = torch.softmax(self.expert_logits, dim=0)
        weights = weights / weights.norm()
        
        z = torch.zeros_like(self.bases[0].flatten())
        
        for i in range(self.n_experts):
            x = self.bases[i].flatten()
            scales = torch.exp(self.expert_log_scales[i].clamp(-2, 2))
            scales = scales / scales.mean()
            x_scaled = x * scales
            z = z + weights[i] * x_scaled
        
        # Normalize
        target_norm = np.sqrt(self.dim)
        z = z * (target_norm / (z.norm() + 1e-9))
        
        return z.reshape(self.noise_shape)


# ══════════════════════════════════════════════════════════════
# Regularization functions (shared)
# ══════════════════════════════════════════════════════════════

def reg_orthogonality(vs: torch.Tensor) -> torch.Tensor:
    """Orthogonality regularization"""
    K, d = vs.shape
    VVt = vs @ vs.t()
    I_K = torch.eye(K, device=vs.device, dtype=vs.dtype)
    return ((VVt - I_K) ** 2).mean() / (d ** 2)


def reg_jb(latent: torch.Tensor) -> torch.Tensor:
    """Jarque-Bera surrogate"""
    z = latent.flatten().float()
    mu = z.mean()
    sigma = z.std() + 1e-8
    z_std = (z - mu) / sigma
    S = (z_std ** 3).mean()
    K = (z_std ** 4).mean() - 3.0
    return S ** 2 + (K ** 2) / 4.0


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
    """KS quantile matching"""
    z = latent.flatten().float()
    n = z.shape[0]
    z_sorted, _ = torch.sort(z)
    indices = torch.linspace(0, n - 1, n_quantiles, device=z.device).long()
    empirical_q = z_sorted[indices]
    theory_q = _get_theory_quantiles(n_quantiles, z.device)
    return ((empirical_q - theory_q) ** 2).mean()


def compute_gaussianity_tests(latent: torch.Tensor) -> dict:
    """Statistical tests for Gaussianity"""
    z = latent.detach().cpu().float().flatten().numpy()
    jb_stat, jb_pvalue = scipy_stats.jarque_bera(z)
    ks_stat, ks_pvalue = scipy_stats.kstest(z, 'norm', args=(0, 1))
    return {
        'jb_pvalue': jb_pvalue,
        'jb_gaussian': jb_pvalue > 0.05,
        'ks_pvalue': ks_pvalue,
        'ks_gaussian': ks_pvalue > 0.05,
    }