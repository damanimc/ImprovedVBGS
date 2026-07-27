"""Colored Ray-space VBGS: spatial NIW + color NIW with latent depth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vbgs.ray_space.depth_posterior import (
    expected_quadratic_energy,
    gaussian_depth_posterior,
    ray_quadratic_coeffs,
    truncated_normal_moments,
)
from vbgs.ray_space.geometry import point_moments, point_on_ray
from vbgs.ray_space.updates import (
    NIWParams,
    energy_const_terms,
    expected_log_pi,
    expected_precision_and_h,
    niw_posterior_from_moments,
    update_dirichlet,
)


@dataclass
class ColoredRayState:
    responsibilities: np.ndarray  # (N, K)
    depth_mean: np.ndarray
    depth_var: np.ndarray
    space: NIWParams
    color: NIWParams
    dirichlet_alpha: np.ndarray


def _niw_loglik(x: np.ndarray, niw: NIWParams) -> np.ndarray:
    """E[log N(x; μ, Σ)] under NIW, shape (N, K)."""
    Lambda, h = expected_precision_and_h(niw)
    hx = x @ h.T
    xLx = np.einsum("ni,kij,nj->nk", x, Lambda, x)
    const = energy_const_terms(niw, Lambda)
    return hx - 0.5 * xLx + const[None, :]


def update_responsibilities_colored(
    C: np.ndarray,
    d: np.ndarray,
    rgb: np.ndarray,
    state: ColoredRayState,
) -> np.ndarray:
    Lambda, h = expected_precision_and_h(state.space)
    a, b = ray_quadratic_coeffs(C, d, Lambda, h)
    const = energy_const_terms(state.space, Lambda)
    space_e = expected_quadratic_energy(
        state.depth_mean, state.depth_var, a, b, const=const[None, :]
    )
    color_e = _niw_loglik(rgb, state.color)
    log_r = expected_log_pi(state.dirichlet_alpha)[None, :] + space_e + color_e
    log_r -= log_r.max(axis=1, keepdims=True)
    r = np.exp(log_r)
    return r / np.maximum(r.sum(axis=1, keepdims=True), 1e-12)


def update_depths_colored(
    C: np.ndarray,
    d: np.ndarray,
    state: ColoredRayState,
    prior_mean: float | np.ndarray,
    prior_precision: float,
    truncate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    Lambda, h = expected_precision_and_h(state.space)
    a, b = ray_quadratic_coeffs(C, d, Lambda, h)
    mean, var = gaussian_depth_posterior(
        state.responsibilities,
        a,
        b,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
    )
    if truncate:
        mean, var = truncated_normal_moments(mean, var, lower=1e-3)
    return mean, var


def fit_colored_ray_vbgs(
    C: np.ndarray,
    d: np.ndarray,
    rgb: np.ndarray,
    *,
    n_components: int = 800,
    n_iters: int = 15,
    depth_prior: np.ndarray | None = None,
    depth_prior_precision: float = 25.0,
    truncate_depth: bool = True,
    fix_depth: bool = False,
    seed: int = 0,
    space_scale: float = 0.05,
    color_scale: float = 0.08,
    temperature: float = 2.0,
) -> ColoredRayState:
    """Batch coordinate ascent for colored RVBGS."""
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    k = n_components
    if depth_prior is None:
        depth_prior = np.full(n, 2.0)
    depth_prior = np.asarray(depth_prior, dtype=np.float64)

    mean_x0 = point_on_ray(C, d, depth_prior)
    # k-means++ style init on xyz|rgb
    feats = np.concatenate([mean_x0, rgb], axis=-1)
    centers = np.empty((k, feats.shape[1]), dtype=np.float64)
    centers[0] = feats[rng.integers(n)]
    closest = np.full(n, np.inf)
    for i in range(1, k):
        dist2 = np.sum((feats - centers[i - 1]) ** 2, axis=-1)
        closest = np.minimum(closest, dist2)
        probs = closest / max(closest.sum(), 1e-12)
        centers[i] = feats[rng.choice(n, p=probs)]
    space_means = centers[:, :3]
    color_means = np.clip(centers[:, 3:], 0.0, 1.0)

    prior_space = NIWParams(
        mean=space_means.copy(),
        kappa=np.full(k, 1.0),
        u=np.array([space_scale * np.eye(3) for _ in range(k)]),
        nu=np.full(k, 6.0),
    )
    prior_color = NIWParams(
        mean=color_means.copy(),
        kappa=np.full(k, 1.0),
        u=np.array([color_scale * np.eye(3) for _ in range(k)]),
        nu=np.full(k, 6.0),
    )
    prior_alpha = np.full(k, 10.0)

    dist2 = np.sum((mean_x0[:, None, :] - space_means[None, :, :]) ** 2, axis=-1)
    col2 = np.sum((rgb[:, None, :] - color_means[None, :, :]) ** 2, axis=-1)
    log_r = -0.5 * (dist2 / max(space_scale, 1e-4) + col2 / max(color_scale, 1e-4))
    log_r -= log_r.max(axis=1, keepdims=True)
    r = np.exp(log_r / max(temperature, 1e-3))
    r /= np.maximum(r.sum(axis=1, keepdims=True), 1e-12)

    state = ColoredRayState(
        responsibilities=r,
        depth_mean=depth_prior.copy(),
        depth_var=np.full(n, 1.0 / max(depth_prior_precision, 1e-6)),
        space=NIWParams(
            mean=space_means.copy(),
            kappa=np.full(k, 1.0),
            u=np.array([space_scale * np.eye(3) for _ in range(k)]),
            nu=np.full(k, 6.0),
        ),
        color=NIWParams(
            mean=color_means.copy(),
            kappa=np.full(k, 1.0),
            u=np.array([color_scale * np.eye(3) for _ in range(k)]),
            nu=np.full(k, 6.0),
        ),
        dirichlet_alpha=prior_alpha.copy(),
    )

    for it in range(n_iters):
        r = update_responsibilities_colored(C, d, rgb, state)
        if temperature != 1.0:
            log_r = np.log(np.maximum(r, 1e-12)) / temperature
            log_r -= log_r.max(axis=1, keepdims=True)
            r = np.exp(log_r)
            r /= np.maximum(r.sum(axis=1, keepdims=True), 1e-12)
        state.responsibilities = r
        if not fix_depth:
            dm, dv = update_depths_colored(
                C,
                d,
                state,
                prior_mean=depth_prior,
                prior_precision=depth_prior_precision,
                truncate=truncate_depth,
            )
            state.depth_mean, state.depth_var = dm, dv
        mean_x, cov_x = point_moments(C, d, state.depth_mean, state.depth_var)
        state.space = niw_posterior_from_moments(mean_x, cov_x, r, prior_space)
        zero_cov = np.zeros((n, 3, 3))
        state.color = niw_posterior_from_moments(rgb, zero_cov, r, prior_color)
        state.dirichlet_alpha = update_dirichlet(prior_alpha, r)
        if (it + 1) % 5 == 0 or it == 0:
            n_eff = float((r.sum(axis=0) > 1.0).sum())
            print(
                f"  iter {it+1}/{n_iters}  active≈{n_eff:.0f}  "
                f"depth_med={np.median(state.depth_mean):.3f}"
            )
    return state


def state_to_model_dict(state: ColoredRayState) -> dict:
    """Export VBGS-compatible model_final.json fields."""
    k = state.space.mean.shape[0]
    mu = np.concatenate([state.space.mean, state.color.mean], axis=-1)
    si = np.zeros((k, 6, 6), dtype=np.float64)
    # Posterior expected covariance ≈ U / (ν - d - 1) for NIW; use U/ν fallback.
    for i in range(k):
        nu_s = max(state.space.nu[i] - 3 - 1, 1.0)
        nu_c = max(state.color.nu[i] - 3 - 1, 1.0)
        si[i, :3, :3] = state.space.u[i] / nu_s
        si[i, 3:, 3:] = state.color.u[i] / nu_c
    alpha = state.dirichlet_alpha / state.dirichlet_alpha.sum()
    # Keep every component with measurable posterior mass.
    keep = state.dirichlet_alpha > (state.dirichlet_alpha.min() + 1e-9)
    keep = state.dirichlet_alpha > 10.05  # prior α0=10
    if int(keep.sum()) < 16:
        order = np.argsort(-state.dirichlet_alpha)
        keep = np.zeros(k, dtype=bool)
        keep[order[: min(k, max(64, k // 2))]] = True
    mu, si, alpha = mu[keep], si[keep], alpha[keep]
    alpha = alpha / np.maximum(alpha.sum(), 1e-12)
    # Shrink huge spatial covariances for stabler splatting.
    for i in range(len(si)):
        evals, evecs = np.linalg.eigh(si[i, :3, :3])
        evals = np.clip(evals, 1e-6, 0.05)
        si[i, :3, :3] = evecs @ np.diag(evals) @ evecs.T
    return {
        "mu": mu.tolist(),
        "si": si.tolist(),
        "alpha": alpha.tolist(),
        "n_semantic": 0,
    }
