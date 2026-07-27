"""Coordinate-ascent updates for Ray-space VBGS (geometry only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vbgs.ray_space.depth_posterior import (
    expected_quadratic_energy,
    gaussian_depth_posterior,
    ray_quadratic_coeffs,
    truncated_normal_moments,
)
from vbgs.ray_space.geometry import point_moments


@dataclass
class NIWParams:
    """Normal-Inverse-Wishart variational parameters per component."""

    mean: np.ndarray  # (K, 3)
    kappa: np.ndarray  # (K,)
    u: np.ndarray  # (K, 3, 3) scale matrix U
    nu: np.ndarray  # (K,) degrees of freedom


@dataclass
class RayVBGSState:
    """Mean-field state for geometry RVBGS."""

    responsibilities: np.ndarray  # (N, K)
    depth_mean: np.ndarray  # (N,)
    depth_var: np.ndarray  # (N,)
    niw: NIWParams
    dirichlet_alpha: np.ndarray  # (K,)


def expected_precision_and_h(niw: NIWParams) -> tuple[np.ndarray, np.ndarray]:
    """Λ = E[Σ^{-1}] = ν U^{-1}, h = Λ μ  (standard NIW expectations).

    Here ``u`` stores the Wishart scale matrix U with E[Σ^{-1}] = ν U^{-1}
    when parameterised so that the rate/scale matches VBGS ``expected_inv_sigma``.
    We use the common form E[Σ^{-1}] = ν U^{-1} with U = niw.u.
    """
    inv_u = np.linalg.inv(niw.u)
    Lambda = niw.nu[:, None, None] * inv_u
    h = np.einsum("kij,kj->ki", Lambda, niw.mean)
    return Lambda, h


def expected_log_pi(alpha: np.ndarray) -> np.ndarray:
    from scipy.special import digamma

    return digamma(alpha) - digamma(alpha.sum())


def energy_const_terms(niw: NIWParams, Lambda: np.ndarray) -> np.ndarray:
    """x-independent part of E[log N(x; μ, Σ)] for each component.

    c = -½ E[μ^T Σ^{-1} μ] + ½ E[log|Σ^{-1}|] - (D/2) log(2π)

    Using E[μ^T Σ^{-1} μ] = μ^T Λ μ + D/κ  (under NIW; Λ=E[Σ^{-1}]),
    and E[log|Σ^{-1}|] ≈ log|Λ| as a practical approximation for the
    reference implementation (full digamma form available in VBGS MVN).
    """
    d = niw.mean.shape[-1]
    mu_L_mu = np.einsum("ki,kij,kj->k", niw.mean, Lambda, niw.mean)
    mu_term = 0.5 * (mu_L_mu + d / np.maximum(niw.kappa, 1e-12))
    sign, logabsdet = np.linalg.slogdet(Lambda)
    logdet = logabsdet  # approx E[log|Σ^{-1}|]
    return -mu_term + 0.5 * logdet - 0.5 * d * np.log(2.0 * np.pi)


def update_responsibilities(
    C: np.ndarray,
    d: np.ndarray,
    state: RayVBGSState,
) -> np.ndarray:
    Lambda, h = expected_precision_and_h(state.niw)
    a, b = ray_quadratic_coeffs(C, d, Lambda, h)
    const = energy_const_terms(state.niw, Lambda)  # (K,)
    e_ell = expected_quadratic_energy(
        state.depth_mean, state.depth_var, a, b, const=const[None, :]
    )
    log_r = expected_log_pi(state.dirichlet_alpha)[None, :] + e_ell
    log_r = log_r - np.max(log_r, axis=1, keepdims=True)
    r = np.exp(log_r)
    r = r / np.maximum(r.sum(axis=1, keepdims=True), 1e-12)
    return r


def update_depths(
    C: np.ndarray,
    d: np.ndarray,
    state: RayVBGSState,
    prior_mean: float | np.ndarray = 1.0,
    prior_precision: float = 1e-6,
    truncate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    Lambda, h = expected_precision_and_h(state.niw)
    a, b = ray_quadratic_coeffs(C, d, Lambda, h)
    mean, var = gaussian_depth_posterior(
        state.responsibilities,
        a,
        b,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
    )
    if truncate:
        mean, var = truncated_normal_moments(mean, var, lower=0.0)
    return mean, var

def niw_posterior_from_moments(
    mean_x: np.ndarray,
    cov_x: np.ndarray,
    responsibilities: np.ndarray,
    prior: NIWParams,
) -> NIWParams:
    """Conjugate NIW update from E[x], Cov(x) and responsibilities."""
    r = responsibilities
    n_k = r.sum(axis=0)  # (K,)
    # weighted mean of E[x]
    sum_x = r.T @ mean_x  # (K, 3)
    xbar = sum_x / np.maximum(n_k[:, None], 1e-12)

    # S_k = Σ_n r_nk [Cov(x_n) + (E[x_n]-x̄)(E[x_n]-x̄)^T]
    diff = mean_x[:, None, :] - xbar[None, :, :]  # (N, K, 3)
    outer = diff[:, :, :, None] * diff[:, :, None, :]  # (N, K, 3, 3)
    scatter = np.einsum("nk,nkij->kij", r, outer)
    scatter = scatter + np.einsum("nk,nij->kij", r, cov_x)

    kappa = prior.kappa + n_k
    nu = prior.nu + n_k
    mean = (
        prior.kappa[:, None] * prior.mean + sum_x
    ) / np.maximum(kappa[:, None], 1e-12)

    # U_post = U_0 + S + (κ0 n)/(κ0+n) (x̄ - m0)(x̄ - m0)^T
    delta = xbar - prior.mean
    prior_scatter = (
        (prior.kappa * n_k)
        / np.maximum(prior.kappa + n_k, 1e-12)
    )[:, None, None] * (delta[:, :, None] * delta[:, None, :])
    u = prior.u + scatter + prior_scatter
    return NIWParams(mean=mean, kappa=kappa, u=u, nu=nu)


def update_dirichlet(prior_alpha: np.ndarray, responsibilities: np.ndarray) -> np.ndarray:
    return prior_alpha + responsibilities.sum(axis=0)


def coordinate_ascent_step(
    C: np.ndarray,
    d: np.ndarray,
    state: RayVBGSState,
    prior_niw: NIWParams,
    prior_alpha: np.ndarray,
    depth_prior_mean: float | np.ndarray = 1.0,
    depth_prior_precision: float = 1e-6,
    truncate_depth: bool = False,
    fix_depth: bool = False,
) -> RayVBGSState:
    """One full coordinate-ascent sweep: q(Z) → q(Λ) → q(Θ)."""
    r = update_responsibilities(C, d, state)
    state = RayVBGSState(
        responsibilities=r,
        depth_mean=state.depth_mean,
        depth_var=state.depth_var,
        niw=state.niw,
        dirichlet_alpha=state.dirichlet_alpha,
    )
    if not fix_depth:
        depth_mean, depth_var = update_depths(
            C,
            d,
            state,
            prior_mean=depth_prior_mean,
            prior_precision=depth_prior_precision,
            truncate=truncate_depth,
        )
    else:
        depth_mean, depth_var = state.depth_mean, state.depth_var

    mean_x, cov_x = point_moments(C, d, depth_mean, depth_var)
    niw = niw_posterior_from_moments(mean_x, cov_x, r, prior_niw)
    alpha = update_dirichlet(prior_alpha, r)
    return RayVBGSState(
        responsibilities=r,
        depth_mean=depth_mean,
        depth_var=depth_var,
        niw=niw,
        dirichlet_alpha=alpha,
    )


def fit_ray_vbgs(
    C: np.ndarray,
    d: np.ndarray,
    n_components: int,
    n_iters: int = 25,
    seed: int = 0,
    depth_init: np.ndarray | None = None,
    depth_prior_mean: float | np.ndarray = 1.0,
    depth_prior_precision: float = 1e-4,
    truncate_depth: bool = True,
    fix_depth: bool = False,
    prior_scale: float = 0.05,
) -> RayVBGSState:
    """Run coordinate ascent from a random NIW initialisation.

    ``depth_prior_mean`` may be a scalar or per-ray array. An informative
    depth prior (or ``fix_depth=True``) is required for well-posed monocular
    geometry; see ``docs/theory/ray_space_vbgs.md``.
    """
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    k = n_components
    if depth_init is None:
        if np.ndim(depth_prior_mean) == 0:
            depth_init = np.full(n, float(depth_prior_mean), dtype=np.float64)
        else:
            depth_init = np.asarray(depth_prior_mean, dtype=np.float64).copy()
    mean_x0 = C + depth_init[:, None] * d
    idx = rng.choice(n, size=k, replace=False)
    means = mean_x0[idx] + 0.05 * rng.normal(size=(k, 3))
    prior_niw = NIWParams(
        mean=np.zeros((k, 3)),
        kappa=np.full(k, 0.1),
        u=np.array([prior_scale * np.eye(3) for _ in range(k)]),
        nu=np.full(k, 5.0),
    )
    # Posterior starts at prior + weak pull toward sampled means.
    init_niw = NIWParams(
        mean=means.copy(),
        kappa=np.full(k, 1.0),
        u=np.array([prior_scale * np.eye(3) for _ in range(k)]),
        nu=np.full(k, 5.0),
    )
    prior_alpha = np.full(k, 1.0)
    # Softmax init by distance to sampled means.
    dist2 = np.sum((mean_x0[:, None, :] - means[None, :, :]) ** 2, axis=-1)
    log_r = -0.5 * dist2 / max(prior_scale, 1e-3)
    log_r -= log_r.max(axis=1, keepdims=True)
    r = np.exp(log_r)
    r /= np.maximum(r.sum(axis=1, keepdims=True), 1e-12)
    state = RayVBGSState(
        responsibilities=r,
        depth_mean=depth_init.astype(np.float64),
        depth_var=np.full(n, 1.0 / max(depth_prior_precision, 1e-6)),
        niw=init_niw,
        dirichlet_alpha=prior_alpha.copy(),
    )
    for _ in range(n_iters):
        state = coordinate_ascent_step(
            C,
            d,
            state,
            prior_niw=prior_niw,
            prior_alpha=prior_alpha,
            depth_prior_mean=depth_prior_mean,
            depth_prior_precision=depth_prior_precision,
            truncate_depth=truncate_depth,
            fix_depth=fix_depth,
        )
    return state
