"""Ray geometry for Ray-space VBGS."""

from __future__ import annotations

import numpy as np


def rays_from_c2w(
    uv: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build camera centres and world directions from camera-to-world poses.

    Args:
        uv: Pixel coordinates, shape (N, 2).
        K: Intrinsics (3, 3).
        c2w: Camera-to-world matrices. Shape (N, 4, 4) or broadcastable (4, 4).

    Returns:
        C: Camera centres in world frame, shape (N, 3).
        d: Viewing directions in world frame, shape (N, 3). Not necessarily unit.
    """
    uv = np.asarray(uv, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    c2w = np.asarray(c2w, dtype=np.float64)
    n = uv.shape[0]

    if c2w.ndim == 2:
        c2w = np.broadcast_to(c2w, (n, 4, 4))

    ones = np.ones((n, 1), dtype=np.float64)
    u_tilde = np.concatenate([uv, ones], axis=-1)  # (N, 3)
    K_inv = np.linalg.inv(K)
    d_cam = u_tilde @ K_inv.T  # (N, 3)
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]
    d_world = np.einsum("nij,nj->ni", R, d_cam)
    return t.copy(), d_world


def rays_from_w2c(
    uv: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build centres/directions from world-to-camera poses.

    Uses C = -R^T t and d = R^T K^{-1} ũ (not C = -t).
    """
    uv = np.asarray(uv, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    w2c = np.asarray(w2c, dtype=np.float64)
    n = uv.shape[0]
    if w2c.ndim == 2:
        w2c = np.broadcast_to(w2c, (n, 4, 4))

    ones = np.ones((n, 1), dtype=np.float64)
    u_tilde = np.concatenate([uv, ones], axis=-1)
    K_inv = np.linalg.inv(K)
    d_cam = u_tilde @ K_inv.T
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    R_t = np.transpose(R, (0, 2, 1))
    C = -np.einsum("nij,nj->ni", R_t, t)
    d_world = np.einsum("nij,nj->ni", R_t, d_cam)
    return C, d_world


def point_on_ray(C: np.ndarray, d: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """x = C + λ d."""
    lam = np.asarray(lam, dtype=np.float64)[..., None]
    return C + lam * d


def point_moments(
    C: np.ndarray, d: np.ndarray, mean_lam: np.ndarray, var_lam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """E[x] and Cov(x) = Var(λ) d d^T under a scalar depth posterior."""
    mean_x = point_on_ray(C, d, mean_lam)
    # Cov_n = var_n * d_n d_n^T  -> (N, 3, 3)
    cov_x = var_lam[:, None, None] * d[:, :, None] * d[:, None, :]
    return mean_x, cov_x
