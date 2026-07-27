"""CPU alpha-blended Gaussian splat renderer for RVBGS eval (no CUDA)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_model(path: Path | str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {
        "mu": np.asarray(raw["mu"], dtype=np.float64),
        "si": np.asarray(raw["si"], dtype=np.float64),
        "alpha": np.asarray(raw["alpha"], dtype=np.float64),
    }


def _focal_from_angle(width: int, height: int, camera_angle_x: float) -> tuple[float, float]:
    fx = height / (2.0 * np.tan(camera_angle_x / 2.0))
    return fx, fx


def render_view(
    model: dict,
    c2w_opengl: np.ndarray,
    camera_angle_x: float,
    width: int,
    height: int,
    bg: float = 0.0,
    opacity_scale: float = 60.0,
) -> np.ndarray:
    """Render RGB image in [0,1] via projected anisotropic Gaussians.

    Uses OpenGL c2w from transforms_*.json (y-up, -z forward). Converts to
    CV axes for projection, sorts by depth, front-to-back alpha composites.
    """
    mu = model["mu"]
    si = model["si"]
    alpha = model["alpha"]
    xyz = mu[:, :3]
    rgb = np.clip(mu[:, 3:6], 0.0, 1.0)
    cov = si[:, :3, :3]

    opengl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = c2w_opengl @ opengl_to_cv
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    Xc = (R @ xyz.T).T + t
    z = Xc[:, 2]
    valid = z > 1e-3
    if not np.any(valid):
        return np.full((height, width, 3), bg, dtype=np.float64)

    fx, fy = _focal_from_angle(width, height, camera_angle_x)
    cx, cy = width / 2.0, height / 2.0
    u = fx * (Xc[:, 0] / z) + cx
    v = fy * (Xc[:, 1] / z) + cy

    # Project covariance to image (Jacobians of pinhole)
    # Σ_img ≈ J R Σ R^T J^T, J = [[fx/z,0,-fx x/z^2],[0,fy/z,-fy y/z^2]]
    order = np.argsort(z)
    img = np.zeros((height, width, 3), dtype=np.float64)
    transm = np.ones((height, width), dtype=np.float64)

    for i in order:
        if not valid[i]:
            continue
        zi = z[i]
        ui, vi = u[i], v[i]
        if ui < -20 or vi < -20 or ui > width + 20 or vi > height + 20:
            continue
        Jc = np.array(
            [
                [fx / zi, 0.0, -fx * Xc[i, 0] / (zi * zi)],
                [0.0, fy / zi, -fy * Xc[i, 1] / (zi * zi)],
            ]
        )
        cov_cam = R @ cov[i] @ R.T
        cov_img = Jc @ cov_cam @ Jc.T
        # stabilize
        cov_img = cov_img + np.eye(2) * 1.0
        try:
            prec = np.linalg.inv(cov_img)
        except np.linalg.LinAlgError:
            continue
        # radius from eigenvalues
        evals = np.linalg.eigvalsh(cov_img)
        rad = int(np.ceil(3.0 * np.sqrt(max(evals.max(), 1e-6))))
        rad = int(np.clip(rad, 2, 50))
        x0 = max(int(np.floor(ui - rad)), 0)
        x1 = min(int(np.ceil(ui + rad)) + 1, width)
        y0 = max(int(np.floor(vi - rad)), 0)
        y1 = min(int(np.ceil(vi + rad)) + 1, height)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        xx, yy = np.meshgrid(xs, ys)
        dx = xx - ui
        dy = yy - vi
        mah = (
            prec[0, 0] * dx * dx
            + 2 * prec[0, 1] * dx * dy
            + prec[1, 1] * dy * dy
        )
        wmap = np.exp(-0.5 * np.clip(mah, 0.0, 50.0))
        opac = float(
            np.clip(1.0 - np.exp(-opacity_scale * max(alpha[i], 1e-4)), 0.05, 0.99)
        )
        a = opac * wmap
        tview = transm[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] + (a * tview)[..., None] * rgb[i]
        transm[y0:y1, x0:x1] = tview * (1.0 - a)

    img = img + transm[..., None] * bg
    return np.clip(img, 0.0, 1.0)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2))
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def eval_split_psnr(
    model_path: Path | str,
    data_path: Path | str,
    split: str = "val",
    *,
    n_frames: int | None = 20,
    image_scale: float = 0.25,
    save_dir: Path | None = None,
) -> dict:
    data_path = Path(data_path)
    model = load_model(model_path)
    with open(data_path / f"transforms_{split}.json") as f:
        meta = json.load(f)
    frames = meta["frames"]
    if n_frames is not None:
        frames = frames[:n_frames]
    angle = float(meta["camera_angle_x"])
    values = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        rgb_path = data_path / f"{frame['file_path']}.png"
        gt_img = Image.open(rgb_path).convert("RGB")
        w0, h0 = gt_img.size
        w = max(1, int(round(w0 * image_scale)))
        h = max(1, int(round(h0 * image_scale)))
        gt = np.asarray(gt_img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        pred = render_view(model, c2w, angle, w, h, bg=0.0)
        val = psnr(pred, gt)
        values.append(val)
        if save_dir is not None:
            canvas = np.concatenate([gt, pred], axis=1)
            Image.fromarray((canvas * 255).astype(np.uint8)).save(
                save_dir / f"{i:03d}_{Path(frame['file_path']).name}_gt_pred.png"
            )
        print(f"  eval {i+1}/{len(frames)} PSNR={val:.3f}")

    return {
        "split": split,
        "n": len(values),
        "mean_psnr": float(np.mean(values)),
        "std_psnr": float(np.std(values)),
        "min_psnr": float(np.min(values)),
        "max_psnr": float(np.max(values)),
        "values": values,
        "image_scale": image_scale,
        "renderer": "cpu_ewa_alpha",
    }
