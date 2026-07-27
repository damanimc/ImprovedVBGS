"""Closed-form depth posterior q(λ) for Ray-space VBGS."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def ray_quadratic_coeffs(
    C: np.ndarray,
    d: np.ndarray,
    Lambda: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic coefficients of ℓ_k(C + λ d) = -½ a λ² + b λ + const.

    Args:
        C: (N, 3) camera centres.
        d: (N, 3) ray directions.
        Lambda: (K, 3, 3) expected precisions E[Σ^{-1}].
        h: (K, 3) expected E[Σ^{-1} μ].

    Returns:
        a: (N, K) with a_{nk} = d_n^T Λ_k d_n.
        b: (N, K) with b_{nk} = h_k^T d_n - C_n^T Λ_k d_n.
    """
    # d^T Λ d: (N, K)
    Ld = np.einsum("kij,nj->nki", Lambda, d)  # (N, K, 3)
    a = np.einsum("nki,ni->nk", Ld, d)
    # h^T d: (N, K)
    hd = np.einsum("ki,ni->nk", h, d)
    # C^T Λ d: (N, K)
    CLd = np.einsum("ni,nki->nk", C, Ld)
    b = hd - CLd
    return a, b


def gaussian_depth_posterior(
    responsibilities: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    prior_mean: float | np.ndarray = 1.0,
    prior_precision: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal unrestricted Gaussian q(λ_n) = N(B_n/A_n, 1/A_n).

    A_n = τ_λ + Σ_k r_{nk} a_{nk}
    B_n = τ_λ m_λ + Σ_k r_{nk} b_{nk}

    ``prior_mean`` may be scalar or shape (N,).
    """
    r = responsibilities
    m0 = np.asarray(prior_mean, dtype=np.float64)
    A = prior_precision + np.sum(r * a, axis=1)
    B = prior_precision * m0 + np.sum(r * b, axis=1)
    A = np.maximum(A, 1e-12)
    mean = B / A
    var = 1.0 / A
    return mean, var

def truncated_normal_moments(
    mean: np.ndarray, var: np.ndarray, lower: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/variance of N(μ, σ²) truncated to (lower, ∞)."""
    sigma = np.sqrt(np.maximum(var, 1e-18))
    alpha = (lower - mean) / sigma
    # Mills ratio φ(α)/(1-Φ(α))
    log_sf = norm.logsf(alpha)
    log_pdf = norm.logpdf(alpha)
    # clip for numerical stability in deep tails
    mills = np.exp(np.clip(log_pdf - log_sf, -50.0, 50.0))
    mean_t = mean + sigma * mills
    var_t = var * (1.0 - mills * (mills - alpha))
    var_t = np.maximum(var_t, 0.0)
    return mean_t, var_t


def expected_quadratic_energy(
    mean_lam: np.ndarray,
    var_lam: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    const: np.ndarray | None = None,
) -> np.ndarray:
    """E_λ[-½ a λ² + b λ + const] for Gaussian depth moments.

    Returns array shaped (N, K).
    """
    m = mean_lam[:, None]
    v = var_lam[:, None]
    # E[λ²] = v + m²
    e_lam2 = v + m * m
    energy = -0.5 * a * e_lam2 + b * m
    if const is not None:
        energy = energy + const
    return energy
