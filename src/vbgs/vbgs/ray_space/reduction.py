"""Proposition 1: Dirac / zero-variance depth reduces RVBGS to VBGS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vbgs.ray_space.depth_posterior import (
    expected_quadratic_energy,
    ray_quadratic_coeffs,
)
from vbgs.ray_space.geometry import point_moments, point_on_ray
from vbgs.ray_space.updates import (
    NIWParams,
    RayVBGSState,
    energy_const_terms,
    expected_log_pi,
    expected_precision_and_h,
    niw_posterior_from_moments,
    update_responsibilities,
)


@dataclass
class ReductionCheck:
    resp_max_abs: float
    mean_max_abs: float
    cov_max_abs: float
    scatter_max_abs: float
    ok: bool


def vbgs_responsibilities(
    x: np.ndarray,
    niw: NIWParams,
    alpha: np.ndarray,
) -> np.ndarray:
    """Standard VBGS (complete-data) responsibility scores."""
    Lambda, h = expected_precision_and_h(niw)
    # ℓ_k(x) = h^T x - ½ x^T Λ x + c
    hx = x @ h.T  # (N, K)
    xLx = np.einsum("ni,kij,nj->nk", x, Lambda, x)
    const = energy_const_terms(niw, Lambda)
    log_r = expected_log_pi(alpha)[None, :] + hx - 0.5 * xLx + const[None, :]
    log_r = log_r - np.max(log_r, axis=1, keepdims=True)
    r = np.exp(log_r)
    return r / np.maximum(r.sum(axis=1, keepdims=True), 1e-12)


def vbgs_scatter(
    x: np.ndarray, responsibilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete-data N_k, x̄_k, S_k."""
    r = responsibilities
    n_k = r.sum(axis=0)
    sum_x = r.T @ x
    xbar = sum_x / np.maximum(n_k[:, None], 1e-12)
    diff = x[:, None, :] - xbar[None, :, :]
    outer = diff[:, :, :, None] * diff[:, :, None, :]
    scatter = np.einsum("nk,nkij->kij", r, outer)
    return n_k, xbar, scatter


def check_proposition_1(
    C: np.ndarray,
    d: np.ndarray,
    depths: np.ndarray,
    niw: NIWParams,
    alpha: np.ndarray,
    atol: float = 1e-8,
) -> ReductionCheck:
    """Verify Dirac depth posterior makes RVBGS updates match VBGS."""
    n = C.shape[0]
    x = point_on_ray(C, d, depths)

    state = RayVBGSState(
        responsibilities=np.full((n, niw.mean.shape[0]), 1.0 / niw.mean.shape[0]),
        depth_mean=depths.copy(),
        depth_var=np.zeros(n),
        niw=niw,
        dirichlet_alpha=alpha,
    )

    # Responsibilities
    r_ray = update_responsibilities(C, d, state)
    r_vbgs = vbgs_responsibilities(x, niw, alpha)
    resp_err = float(np.max(np.abs(r_ray - r_vbgs)))

    # Moments
    mean_x, cov_x = point_moments(C, d, depths, np.zeros(n))
    mean_err = float(np.max(np.abs(mean_x - x)))
    cov_err = float(np.max(np.abs(cov_x)))

    # Scatter / NIW stats — compare posterior under a shared prior.
    prior = NIWParams(
        mean=niw.mean.copy(),
        kappa=np.ones_like(niw.kappa),
        u=np.array([np.eye(3) for _ in range(niw.mean.shape[0])]),
        nu=np.full(niw.mean.shape[0], 5.0),
    )
    post_ray = niw_posterior_from_moments(mean_x, cov_x, r_vbgs, prior)
    n_k, xbar, s_vbgs = vbgs_scatter(x, r_vbgs)
    # complete-data NIW update
    kappa = prior.kappa + n_k
    sum_x = r_vbgs.T @ x
    mean_vbgs = (prior.kappa[:, None] * prior.mean + sum_x) / kappa[:, None]
    delta = xbar - prior.mean
    prior_scatter = (
        (prior.kappa * n_k) / (prior.kappa + n_k)
    )[:, None, None] * (delta[:, :, None] * delta[:, None, :])
    u_vbgs = prior.u + s_vbgs + prior_scatter

    scatter_err = float(
        max(
            np.max(np.abs(post_ray.mean - mean_vbgs)),
            np.max(np.abs(post_ray.u - u_vbgs)),
            np.max(np.abs(post_ray.kappa - kappa)),
        )
    )

    # Also check expected energy identity at zero variance
    Lambda, h = expected_precision_and_h(niw)
    a, b = ray_quadratic_coeffs(C, d, Lambda, h)
    e_quad = expected_quadratic_energy(depths, np.zeros(n), a, b)
    hx = x @ h.T
    xLx = np.einsum("ni,kij,nj->nk", x, Lambda, x)
    e_point = hx - 0.5 * xLx
    energy_err = float(np.max(np.abs(e_quad - e_point)))

    ok = max(resp_err, mean_err, cov_err, scatter_err, energy_err) < atol
    return ReductionCheck(
        resp_max_abs=resp_err,
        mean_max_abs=max(mean_err, energy_err),
        cov_max_abs=cov_err,
        scatter_max_abs=scatter_err,
        ok=ok,
    )


def ray_uncertainty_inflates_scatter(
    C: np.ndarray,
    d: np.ndarray,
    depths: np.ndarray,
    depth_vars: np.ndarray,
    responsibilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (S_dirac, S_uncertain); uncertain should dominate along rays."""
    x = point_on_ray(C, d, depths)
    _, _, s_dirac = vbgs_scatter(x, responsibilities)
    mean_x, cov_x = point_moments(C, d, depths, depth_vars)
    # scatter with cov
    r = responsibilities
    n_k = r.sum(axis=0)
    sum_x = r.T @ mean_x
    xbar = sum_x / np.maximum(n_k[:, None], 1e-12)
    diff = mean_x[:, None, :] - xbar[None, :, :]
    outer = diff[:, :, :, None] * diff[:, :, None, :]
    s_unc = np.einsum("nk,nkij->kij", r, outer) + np.einsum(
        "nk,nij->kij", r, cov_x
    )
    return s_dirac, s_unc
