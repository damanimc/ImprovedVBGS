"""Numerical verification of Ray-space VBGS identities and Proposition 1."""

from __future__ import annotations

import numpy as np
import pytest

from vbgs.ray_space import (
    NIWParams,
    check_proposition_1,
    fit_ray_vbgs,
    gaussian_depth_posterior,
    point_on_ray,
    ray_quadratic_coeffs,
    ray_uncertainty_inflates_scatter,
    rays_from_c2w,
    rays_from_w2c,
    truncated_normal_moments,
)


def _synthetic_scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    means = np.array([[-0.5, 0.0, 3.0], [0.6, 0.1, 4.0]])
    cov = 0.05 * np.eye(3)
    n_per = 80
    pts = []
    labels = []
    for k, m in enumerate(means):
        pts.append(rng.multivariate_normal(m, cov, size=n_per))
        labels.append(np.full(n_per, k))
    x = np.concatenate(pts, axis=0)
    labels = np.concatenate(labels)

    K = np.array([[200.0, 0.0, 64.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    x_cam = x
    uv = (K @ x_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    depths = x_cam[:, 2]
    C, d = rays_from_c2w(uv, K, c2w)
    x_rec = point_on_ray(C, d, depths)
    assert np.allclose(x_rec, x, atol=1e-8)
    return C, d, depths, x, labels, means


def test_c2w_w2c_consistency():
    rng = np.random.default_rng(1)
    uv = rng.uniform(0, 128, size=(20, 2))
    K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1.0]])
    from scipy.spatial.transform import Rotation

    R = Rotation.random(random_state=1).as_matrix()
    t = rng.normal(size=3)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = np.linalg.inv(c2w)
    C1, d1 = rays_from_c2w(uv, K, c2w)
    C2, d2 = rays_from_w2c(uv, K, w2c)
    assert np.allclose(C1, C2, atol=1e-10)
    assert np.allclose(d1, d2, atol=1e-10)
    assert not np.allclose(C2, -w2c[:3, 3])


def test_proposition_1_reduction():
    C, d, depths, x, _, _ = _synthetic_scene()
    k = 2
    niw = NIWParams(
        mean=np.array([[-0.5, 0.0, 3.0], [0.6, 0.1, 4.0]]),
        kappa=np.ones(k),
        u=np.array([0.1 * np.eye(3) for _ in range(k)]),
        nu=np.full(k, 8.0),
    )
    alpha = np.ones(k)
    result = check_proposition_1(C, d, depths, niw, alpha, atol=1e-8)
    assert result.ok, result


def test_depth_posterior_closed_form_matches_grid():
    """Optimal Gaussian q(λ) matches 1D score maximisation on a grid."""
    C = np.array([[0.0, 0.0, 0.0]])
    d = np.array([[0.0, 0.0, 1.0]])
    Lambda = np.array([np.eye(3)])
    h = np.array([[0.0, 0.0, 3.0]])
    a, b = ray_quadratic_coeffs(C, d, Lambda, h)
    r = np.array([[1.0]])
    mean, var = gaussian_depth_posterior(
        r, a, b, prior_mean=0.0, prior_precision=0.0
    )
    assert mean[0] == pytest.approx(3.0, abs=1e-10)
    assert var[0] == pytest.approx(1.0, abs=1e-10)

    grid = np.linspace(0.5, 5.5, 401)
    score = -0.5 * a[0, 0] * grid**2 + b[0, 0] * grid
    assert grid[np.argmax(score)] == pytest.approx(mean[0], abs=0.02)


def test_ray_uncertainty_inflates_scatter():
    C, d, depths, _, _, _ = _synthetic_scene()
    n, k = C.shape[0], 2
    r = np.full((n, k), 1.0 / k)
    s0, s1 = ray_uncertainty_inflates_scatter(
        C, d, depths, np.full(n, 0.25), r
    )
    assert np.linalg.norm(s1) > np.linalg.norm(s0)
    extra = s1 - s0
    for mat in extra:
        eig = np.linalg.eigvalsh(mat)
        assert np.all(eig >= -1e-10)


def test_truncated_moments_positive():
    mean = np.array([-1.0, 0.5, 2.0])
    var = np.array([1.0, 1.0, 0.25])
    m, v = truncated_normal_moments(mean, var, lower=0.0)
    assert np.all(m > 0)
    assert np.all(v >= 0)
    assert m[0] > mean[0]


def test_noisy_depth_prior_refines_geometry():
    """Informative depth prior (monocular prior) + CA recovers blob centres."""
    rng = np.random.default_rng(4)
    C, d, depths, x, _, means = _synthetic_scene(seed=4)
    noisy = depths + rng.normal(0.0, 0.35, size=depths.shape)
    noisy = np.maximum(noisy, 0.2)
    state = fit_ray_vbgs(
        C,
        d,
        n_components=2,
        n_iters=40,
        seed=4,
        depth_init=noisy,
        depth_prior_mean=noisy,
        depth_prior_precision=4.0,
        truncate_depth=True,
    )
    x_hat = point_on_ray(C, d, state.depth_mean)
    # Should improve on the noisy prior completion.
    err_prior = np.linalg.norm(point_on_ray(C, d, noisy) - x, axis=1)
    err_fit = np.linalg.norm(x_hat - x, axis=1)
    assert np.median(err_fit) < np.median(err_prior)
    assert np.median(err_fit) < 0.25
    est = state.niw.mean
    d00 = np.linalg.norm(est[0] - means[0])
    d01 = np.linalg.norm(est[0] - means[1])
    d10 = np.linalg.norm(est[1] - means[0])
    d11 = np.linalg.norm(est[1] - means[1])
    assert min(d00 + d11, d01 + d10) < 0.8


def test_fixed_depth_matches_complete_data_means():
    """With depths fixed (Prop 1 regime), NIW means track complete-data VBGS."""
    C, d, depths, x, _, means = _synthetic_scene(seed=3)
    state = fit_ray_vbgs(
        C,
        d,
        n_components=2,
        n_iters=30,
        seed=3,
        depth_init=depths.copy(),
        truncate_depth=False,
        fix_depth=True,
        depth_prior_precision=1e6,
    )
    est = state.niw.mean
    d00 = np.linalg.norm(est[0] - means[0])
    d01 = np.linalg.norm(est[0] - means[1])
    d10 = np.linalg.norm(est[1] - means[0])
    d11 = np.linalg.norm(est[1] - means[1])
    assert min(d00 + d11, d01 + d10) < 0.5
