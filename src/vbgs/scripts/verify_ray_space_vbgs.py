#!/usr/bin/env python3
"""Run Ray-space VBGS verification suite and print a short report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vbgs.ray_space import (  # noqa: E402
    NIWParams,
    check_proposition_1,
    fit_ray_vbgs,
    gaussian_depth_posterior,
    point_on_ray,
    ray_quadratic_coeffs,
    ray_uncertainty_inflates_scatter,
    rays_from_c2w,
    rays_from_w2c,
)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    rng = np.random.default_rng(0)
    failures = 0

    section("Camera convention: c2w ↔ w2c")
    uv = rng.uniform(0, 128, size=(32, 2))
    K = np.array([[120.0, 0, 64], [0, 120.0, 64], [0, 0, 1.0]])
    from scipy.spatial.transform import Rotation

    R = Rotation.random(random_state=0).as_matrix()
    t = rng.normal(size=3)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = np.linalg.inv(c2w)
    C1, d1 = rays_from_c2w(uv, K, c2w)
    C2, d2 = rays_from_w2c(uv, K, w2c)
    ok_pose = np.allclose(C1, C2) and np.allclose(d1, d2)
    wrong_draft = np.allclose(C2, -w2c[:3, 3])
    print(f"c2w/w2c agree: {ok_pose}")
    print(f"draft C=-t holds (should be False for random R): {wrong_draft}")
    failures += int(not ok_pose) + int(wrong_draft)

    section("Proposition 1: Dirac depth → VBGS")
    means = np.array([[-0.4, 0.0, 2.5], [0.5, 0.0, 3.5]])
    pts = np.concatenate(
        [
            rng.multivariate_normal(means[0], 0.04 * np.eye(3), 60),
            rng.multivariate_normal(means[1], 0.04 * np.eye(3), 60),
        ]
    )
    K = np.array([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1.0]])
    c2w = np.eye(4)
    uv = (K @ pts.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    depths = pts[:, 2]
    C, d = rays_from_c2w(uv, K, c2w)
    niw = NIWParams(
        mean=means.copy(),
        kappa=np.ones(2),
        u=np.array([0.1 * np.eye(3) for _ in range(2)]),
        nu=np.full(2, 8.0),
    )
    result = check_proposition_1(C, d, depths, niw, np.ones(2))
    print(result)
    failures += int(not result.ok)

    section("Proposition 2: closed-form depth mode")
    C0 = np.array([[0.0, 0.0, 0.0]])
    d0 = np.array([[0.0, 0.0, 1.0]])
    Lambda = np.array([np.eye(3)])
    h = np.array([[0.0, 0.0, 3.0]])
    a, b = ray_quadratic_coeffs(C0, d0, Lambda, h)
    mean, var = gaussian_depth_posterior(
        np.array([[1.0]]), a, b, prior_mean=0.0, prior_precision=0.0
    )
    print(f"depth mean={mean[0]:.6f} (expect 3), var={var[0]:.6f} (expect 1)")
    failures += int(abs(mean[0] - 3.0) > 1e-8) + int(abs(var[0] - 1.0) > 1e-8)

    section("Ray uncertainty inflates NIW scatter")
    s0, s1 = ray_uncertainty_inflates_scatter(
        C, d, depths, np.full(len(depths), 0.2), np.full((len(depths), 2), 0.5)
    )
    print(
        f"||S_dirac||={np.linalg.norm(s0):.4f}  ||S_unc||={np.linalg.norm(s1):.4f}"
    )
    failures += int(not (np.linalg.norm(s1) > np.linalg.norm(s0)))

    section("Noisy depth prior + coordinate ascent")
    noisy = np.maximum(depths + rng.normal(0.0, 0.35, size=depths.shape), 0.2)
    state = fit_ray_vbgs(
        C,
        d,
        n_components=2,
        n_iters=40,
        seed=0,
        depth_init=noisy,
        depth_prior_mean=noisy,
        depth_prior_precision=4.0,
        truncate_depth=True,
    )
    x_prior = point_on_ray(C, d, noisy)
    x_hat = point_on_ray(C, d, state.depth_mean)
    med_prior = float(np.median(np.linalg.norm(x_prior - pts, axis=1)))
    med_fit = float(np.median(np.linalg.norm(x_hat - pts, axis=1)))
    print(f"median error prior={med_prior:.4f}  after fit={med_fit:.4f}")
    print(f"component means:\n{state.niw.mean}")
    failures += int(not (med_fit < med_prior and med_fit < 0.25))

    section("Summary")
    if failures == 0:
        print("All checks passed.")
        return 0
    print(f"{failures} check(s) failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
