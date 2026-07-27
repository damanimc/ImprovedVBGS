"""Blender / NeRF-Synthetic ray + RGB(+depth) loading for Ray-space VBGS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vbgs.ray_space.geometry import rays_from_c2w

# NumPy copy of OpenGL→CV axis flip used in vbgs.camera.
_OPENGL_TO_FRAME = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def blender_intrinsics(image_hw: tuple[int, int], camera_angle_x: float) -> np.ndarray:
    """Match VersesTech VBGS BlenderDataIterator K construction."""
    h, w = image_hw
    fx = h / (2.0 * np.tan(camera_angle_x / 2.0))
    fy = fx
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def distance_to_depth_scale(camera_angle_x: float, image_hw: tuple[int, int]) -> np.ndarray:
    """|cos θ| ray-distance → Z scale (VBGS Blender convention, abs-stable)."""
    h, w = image_hw
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    uv = np.stack(
        [
            (uu - w / 2.0) * (camera_angle_x / 2.0),
            (vv - h / 2.0) * (camera_angle_x / 2.0),
            np.ones((h, w), dtype=np.float64),
        ],
        axis=-1,
    )
    uvr = uv.reshape(-1, 3)
    uvr = uvr / np.linalg.norm(uvr, axis=-1, keepdims=True)
    # OpenGL forward is -Z; take absolute cosine for a positive Z scale.
    fwd = np.array([0.0, 0.0, -1.0])
    r = np.abs(uvr @ fwd)
    return r.reshape(h, w)


def decode_blender_depth(
    depth_path: Path, camera_angle_x: float, image_hw: tuple[int, int]
) -> np.ndarray:
    """Decode NeRF-Synthetic Blender depth PNGs.

    Public mirrors store a far-plane normalised channel; VBGS recovers metric
    depth via ``8 * (1 - c/255)``. We skip the legacy ``* r`` cosine factor
    from VersesTech's loader — on these PNGs it collapses depth toward 0.
    ``camera_angle_x`` / ``image_hw`` kept for API compatibility.
    """
    del camera_angle_x, image_hw
    depth_im = np.asarray(Image.open(depth_path), dtype=np.float64)
    ch = depth_im[..., 0] if depth_im.ndim == 3 else depth_im
    depth = 8.0 * (1.0 - (ch / 255.0))
    depth *= ch > 0
    return depth


def load_split_meta(data_path: Path, split: str) -> tuple[dict, list]:
    data_path = Path(data_path)
    with open(data_path / f"transforms_{split}.json") as f:
        meta = json.load(f)
    return meta, meta["frames"]


def load_frame_rays(
    data_path: Path,
    frame: dict,
    camera_angle_x: float,
    *,
    image_scale: float = 0.25,
    max_pixels: int | None = 20_000,
    require_depth: bool = False,
    seed: int = 0,
    bg_threshold: float = 0.98,
) -> dict:
    """Load one Blender frame as rays, colours, optional GT depth prior.

    Returns dict with keys: C, d, rgb, depth_prior, uv, K, c2w, has_depth.
    """
    data_path = Path(data_path)
    rgb_path = data_path / f"{frame['file_path']}.png"
    rgb = np.asarray(Image.open(rgb_path).convert("RGBA"), dtype=np.float64)
    h0, w0 = rgb.shape[:2]
    if image_scale != 1.0:
        new_w = max(1, int(round(w0 * image_scale)))
        new_h = max(1, int(round(h0 * image_scale)))
        rgb_img = Image.fromarray(rgb.astype(np.uint8)).resize(
            (new_w, new_h), Image.BILINEAR
        )
        rgb = np.asarray(rgb_img, dtype=np.float64)
    h, w = rgb.shape[:2]
    rgb01 = rgb[..., :3] / 255.0
    alpha = rgb[..., 3] / 255.0 if rgb.shape[-1] == 4 else np.ones((h, w))

    K = blender_intrinsics((h, w), camera_angle_x)
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    # OpenGL → CV camera axes (same as vbgs.camera.construct_* from_opengl)
    c2w = c2w @ _OPENGL_TO_FRAME

    depth = None
    depth_paths = list(data_path.glob(f"{frame['file_path']}_depth_*.png"))
    if depth_paths:
        depth_full = decode_blender_depth(depth_paths[0], camera_angle_x, (h0, w0))
        if image_scale != 1.0:
            depth_img = Image.fromarray(depth_full.astype(np.float32), mode="F")
            depth = np.asarray(
                depth_img.resize((w, h), Image.NEAREST), dtype=np.float64
            )
        else:
            depth = depth_full
    elif require_depth:
        raise FileNotFoundError(f"No depth for {frame['file_path']}")

    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    mask = alpha > 0.1
    # drop near-white background common in NeRF synthetic
    mask &= rgb01.max(axis=-1) < bg_threshold
    if depth is not None:
        mask &= depth > 1e-4

    uv = np.stack([uu[mask], vv[mask]], axis=-1).astype(np.float64)
    cols = rgb01[mask]
    if depth is not None:
        depth_vals = depth[mask]
    else:
        depth_vals = None

    if max_pixels is not None and uv.shape[0] > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(uv.shape[0], size=max_pixels, replace=False)
        uv = uv[idx]
        cols = cols[idx]
        if depth_vals is not None:
            depth_vals = depth_vals[idx]

    C, d = rays_from_c2w(uv, K, c2w)

    # With x = C + λ d and d = R K^{-1} ũ, Blender Z-depth equals λ
    # (see vbgs.camera.transform_uvd_to_points homogeneous construction).
    if depth_vals is not None:
        lam = np.asarray(depth_vals, dtype=np.float64)
    else:
        # Object-centric prior: closest point on ray to origin
        dd = np.sum(d * d, axis=-1)
        lam = -np.sum(C * d, axis=-1) / np.maximum(dd, 1e-12)
        lam = np.clip(lam, 0.5, 8.0)

    return {
        "C": C,
        "d": d,
        "rgb": cols,
        "depth_prior": lam.astype(np.float64),
        "has_depth": depth_vals is not None,
        "uv": uv,
        "K": K,
        "c2w": c2w,
        "image_hw": (h, w),
        "camera_angle_x": camera_angle_x,
        "file_path": frame["file_path"],
    }


def load_ray_dataset(
    data_path: Path | str,
    split: str = "train",
    *,
    n_frames: int | None = 40,
    image_scale: float = 0.25,
    max_pixels_per_frame: int = 8_000,
    seed: int = 0,
    prefer_depth_split_fallback: bool = True,
) -> dict:
    """Stack rays from a Blender split into one RVBGS batch."""
    data_path = Path(data_path)
    meta, frames = load_split_meta(data_path, split)
    # Train/val often lack depth in public mirrors; fall back to test (has depth).
    if prefer_depth_split_fallback and split in {"train", "val"}:
        probe = load_frame_rays(
            data_path,
            frames[0],
            float(meta["camera_angle_x"]),
            image_scale=image_scale,
            max_pixels=16,
            seed=seed,
        )
        if not probe["has_depth"]:
            meta, frames = load_split_meta(data_path, "test")
            split = "test"

    if n_frames is not None:
        frames = frames[:n_frames]

    bundles = []
    for i, frame in enumerate(frames):
        bundles.append(
            load_frame_rays(
                data_path,
                frame,
                float(meta["camera_angle_x"]),
                image_scale=image_scale,
                max_pixels=max_pixels_per_frame,
                seed=seed + i,
            )
        )

    return {
        "split": split,
        "camera_angle_x": float(meta["camera_angle_x"]),
        "C": np.concatenate([b["C"] for b in bundles], axis=0),
        "d": np.concatenate([b["d"] for b in bundles], axis=0),
        "rgb": np.concatenate([b["rgb"] for b in bundles], axis=0),
        "depth_prior": np.concatenate([b["depth_prior"] for b in bundles], axis=0),
        "frames": bundles,
        "n_frames": len(bundles),
    }
