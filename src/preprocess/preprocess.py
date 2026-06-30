"""Preprocess datasets into VBGS-friendly frame point clouds.

Supported inputs:
- TUM RGB-D folders: writes one ``frame_XXXXXX.npz`` per selected RGB-D frame.
- Video files: extracts RGB frames and writes a manifest. Monocular video still
  needs depth and camera poses before VBGS can train a 3D scene.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


TUM_FREIBURG1_INTRINSICS = {
    "fx": 517.3,
    "fy": 516.5,
    "cx": 318.6,
    "cy": 255.3,
    "depth_scale": 5000.0,
}


@dataclass(frozen=True)
class TumAssociation:
    rgb_time: float
    rgb_path: Path
    depth_time: float
    depth_path: Path
    pose_time: float
    pose: np.ndarray


def _read_tum_list(path: Path) -> list[tuple[float, str]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stamp, value = line.split()[:2]
            rows.append((float(stamp), value))
    return rows


def _read_tum_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            stamp = float(parts[0])
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:8])
            rows.append((stamp, _pose_matrix(tx, ty, tz, qx, qy, qz, qw)))
    return rows


def _pose_matrix(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    w, x, y, z = quat
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = [tx, ty, tz]
    return pose


def _nearest(stamp: float, rows):
    idx = int(np.argmin([abs(stamp - row[0]) for row in rows]))
    return rows[idx]


def associate_tum(tum_path: Path, max_dt: float = 0.02) -> list[TumAssociation]:
    rgb_rows = _read_tum_list(tum_path / "rgb.txt")
    depth_rows = _read_tum_list(tum_path / "depth.txt")
    pose_rows = _read_tum_groundtruth(tum_path / "groundtruth.txt")

    associations = []
    for rgb_time, rgb_rel in rgb_rows:
        depth_time, depth_rel = _nearest(rgb_time, depth_rows)
        pose_time, pose = _nearest(rgb_time, pose_rows)
        if abs(rgb_time - depth_time) <= max_dt and abs(rgb_time - pose_time) <= max_dt:
            associations.append(
                TumAssociation(
                    rgb_time=rgb_time,
                    rgb_path=tum_path / rgb_rel,
                    depth_time=depth_time,
                    depth_path=tum_path / depth_rel,
                    pose_time=pose_time,
                    pose=pose,
                )
            )
    return associations


def tum_frame_to_cloud(
    rgb_path: Path,
    depth_path: Path,
    camera_to_world: np.ndarray,
    intrinsics: dict[str, float],
    subsample: int | None,
    rng: np.random.Generator,
    semantic: bool = False,
    semantic_classes: int | None = None,
    semantic_max_masks: int = 48,
    semantic_device: str = "cuda",
) -> np.ndarray:
    rgb_u8 = np.asarray(Image.open(rgb_path).convert("RGB"))
    rgb = rgb_u8.astype(np.float32) / 255.0
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    z = depth_raw / float(intrinsics["depth_scale"])
    valid = z > 0

    height, width = z.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    points_world = points_cam.reshape(-1, 4) @ camera_to_world.T

    flat_valid = valid.reshape(-1)
    features = [points_world[flat_valid, :3], rgb.reshape(-1, 3)[flat_valid]]
    if semantic:
        from vbgs.semantic import attach_semantic_features

        _, onehot = attach_semantic_features(
            rgb_u8,
            depth_raw,
            num_classes=semantic_classes,
            max_masks=semantic_max_masks,
            device=semantic_device,
        )
        features.append(onehot)
    data = np.concatenate(features, axis=1)
    if subsample is not None and data.shape[0] > subsample:
        idx = rng.choice(data.shape[0], size=subsample, replace=False)
        data = data[idx]
    return data.astype(np.float32)


def preprocess_tum(args: argparse.Namespace) -> None:
    tum_path = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    associations = associate_tum(tum_path)
    selected = associations[:: args.stride]
    if args.frames is not None:
        selected = selected[: args.frames]

    manifest = {
        "type": "tum_rgbd_npz",
        "source": str(tum_path),
        "frames": [],
        "intrinsics": TUM_FREIBURG1_INTRINSICS,
        "subsample": args.subsample,
        "semantic": bool(args.semantic),
        "semantic_classes": args.semantic_classes,
    }
    for frame_idx, assoc in enumerate(selected):
        data = tum_frame_to_cloud(
            assoc.rgb_path,
            assoc.depth_path,
            assoc.pose,
            TUM_FREIBURG1_INTRINSICS,
            args.subsample,
            rng,
            semantic=args.semantic,
            semantic_classes=args.semantic_classes,
            semantic_max_masks=args.semantic_max_masks,
            semantic_device=args.semantic_device,
        )
        out_path = output / f"frame_{frame_idx:06d}.npz"
        np.savez_compressed(out_path, data)
        manifest["frames"].append(
            {
                "file": out_path.name,
                "points": int(data.shape[0]),
                "rgb_time": assoc.rgb_time,
                "depth_time": assoc.depth_time,
                "pose_time": assoc.pose_time,
            }
        )
        print(f"{out_path}: {data.shape[0]} points")

    with (output / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def preprocess_video(args: argparse.Namespace) -> None:
    import cv2

    video = Path(args.input)
    output = Path(args.output)
    rgb_dir = output / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")

    frame_idx = 0
    written = 0
    manifest = {
        "type": "video_frames",
        "source": str(video),
        "requires_depth_and_pose": True,
        "frames": [],
    }
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride == 0:
            out_path = rgb_dir / f"frame_{written:06d}.png"
            cv2.imwrite(str(out_path), frame)
            manifest["frames"].append({"file": str(out_path.relative_to(output)), "source_frame": frame_idx})
            written += 1
            if args.frames is not None and written >= args.frames:
                break
        frame_idx += 1
    cap.release()

    with (output / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {written} frames to {rgb_dir}")
    print("video extraction done; depth maps and camera poses still required for VBGS 3D training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    tum = subparsers.add_parser("tum-rgbd")
    tum.add_argument("--input", required=True)
    tum.add_argument("--output", required=True)
    tum.add_argument("--frames", type=int, default=None)
    tum.add_argument("--stride", type=int, default=1)
    tum.add_argument("--subsample", type=int, default=None)
    tum.add_argument("--seed", type=int, default=0)
    tum.add_argument("--semantic", action="store_true")
    tum.add_argument("--semantic-classes", type=int, default=None)
    tum.add_argument("--semantic-max-masks", type=int, default=48)
    tum.add_argument("--semantic-device", default="cuda")
    tum.set_defaults(func=preprocess_tum)

    video = subparsers.add_parser("video")
    video.add_argument("--input", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--frames", type=int, default=None)
    video.add_argument("--stride", type=int, default=1)
    video.set_defaults(func=preprocess_video)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
