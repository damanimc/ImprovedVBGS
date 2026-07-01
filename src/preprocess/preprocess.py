"""Preprocess datasets into the standard VBGS RGB-D scene format.

Supported inputs:
- TUM RGB-D folders: writes RGB frames, metric depth, and camera poses.
- Blender/NeRF Synthetic folders: copies RGB-D files and transform metadata.
- Video files: writes RGB frames, transforms JSON, and depth/pose slots.
  Depth Anything 3 can estimate both depth and camera poses.
"""

from __future__ import annotations

import argparse
import json
import shutil
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

DEFAULT_CAMERA_ANGLE_X = 0.6911112070083618


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


def _camera_angle_x_from_intrinsics(width: int, fx: float) -> float:
    return float(2 * np.arctan2(width / 2.0, float(fx)))


def _opencv_c2w_to_opengl_c2w(pose: np.ndarray) -> np.ndarray:
    """Convert TUM/OpenCV camera axes to Blender/NeRF transform axes."""
    converted = np.array(pose, dtype=np.float64, copy=True)
    converted[:3, 1:3] *= -1
    return converted


def _write_scene_transforms(
    output: Path,
    name: str,
    frames: list[dict],
    camera_angle_x: float,
    *,
    metadata: dict | None = None,
) -> None:
    payload = {
        "camera_angle_x": float(camera_angle_x),
        "frames": frames,
    }
    if metadata:
        payload.update(metadata)
    with (output / f"transforms_{name}.json").open("w") as f:
        json.dump(payload, f, indent=2)


def _write_scene_manifest(
    output: Path,
    source_type: str,
    source: Path,
    *,
    train_frames: int,
    val_frames: int,
    normalization_ranges: list[list[float]],
    warning: str | None = None,
) -> None:
    manifest = {
        "type": "vbgs_scene",
        "source_type": source_type,
        "source": str(source),
        "train_frames": int(train_frames),
        "val_frames": int(val_frames),
        "normalization_ranges": normalization_ranges,
    }
    if warning:
        manifest["warning"] = warning
    with (output / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def _split_name(index: int, val_stride: int) -> str:
    if val_stride > 0 and index % val_stride == 0:
        return "val"
    return "train"


def preprocess_tum(args: argparse.Namespace) -> None:
    tum_path = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_dir = output / "train"
    val_dir = output / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    associations = associate_tum(tum_path)
    selected = associations[:: args.stride]
    if args.frames is not None:
        selected = selected[: args.frames]

    train_frames = []
    val_frames = []
    for frame_idx, assoc in enumerate(selected):
        split = _split_name(frame_idx, args.val_stride)
        split_dir = val_dir if split == "val" else train_dir
        stem = f"frame_{frame_idx:06d}"
        rgb_path = split_dir / f"{stem}.png"
        Image.open(assoc.rgb_path).convert("RGB").save(rgb_path)

        depth_raw = np.asarray(Image.open(assoc.depth_path), dtype=np.float32)
        depth_m = depth_raw / float(TUM_FREIBURG1_INTRINSICS["depth_scale"])
        np.save(split_dir / f"{stem}_depth_da3.npy", depth_m.astype(np.float32))

        record = {
            "file_path": f"{split}/{stem}",
            "rgb_time": assoc.rgb_time,
            "depth_time": assoc.depth_time,
            "pose_time": assoc.pose_time,
            "transform_matrix": _opencv_c2w_to_opengl_c2w(assoc.pose).tolist(),
        }
        if split == "val":
            val_frames.append(record)
        else:
            train_frames.append(record)
        print(f"{rgb_path}")

    if not train_frames and val_frames:
        train_frames.append(val_frames.pop(0))

    first_rgb = Image.open(selected[0].rgb_path)
    camera_angle_x = _camera_angle_x_from_intrinsics(
        first_rgb.size[0],
        TUM_FREIBURG1_INTRINSICS["fx"],
    )
    metadata = {"intrinsics": TUM_FREIBURG1_INTRINSICS}
    _write_scene_transforms(output, "train", train_frames, camera_angle_x, metadata=metadata)
    _write_scene_transforms(output, "val", val_frames, camera_angle_x, metadata=metadata)
    _write_scene_transforms(output, "test", val_frames or train_frames, camera_angle_x, metadata=metadata)
    _write_scene_manifest(
        output,
        "tum-rgbd",
        tum_path,
        train_frames=len(train_frames),
        val_frames=len(val_frames),
        normalization_ranges=[[-5, 5], [-5, 5], [-5, 5], [0, 1], [0, 1], [0, 1]],
    )


def _stationary_blender_pose() -> list[list[float]]:
    return np.eye(4, dtype=np.float32).tolist()


def _load_pose_json(path: Path) -> dict[str, np.ndarray]:
    """Load frame poses from a Blender-style transforms JSON."""
    with path.open() as f:
        data = json.load(f)
    poses = {}
    for frame in data.get("frames", []):
        file_path = Path(frame["file_path"]).as_posix()
        stem = Path(file_path).stem
        poses[file_path] = np.asarray(frame["transform_matrix"], dtype=np.float64)
        poses[stem] = poses[file_path]
    return poses


def _select_torch_device(device: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Depth Anything 3 requires optional dependencies. Install with "
            '`cd src/vbgs && pip install -e ".[depth]"`.'
        ) from exc

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _w2c_to_opengl_c2w(extrinsic: np.ndarray) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape == (4, 4):
        w2c = extrinsic
    else:
        w2c[:3, :4] = extrinsic[:3, :4]
    c2w = np.linalg.inv(w2c)
    return _opencv_c2w_to_opengl_c2w(c2w)


def _run_depth_anything3(
    image_paths: list[Path],
    stems: list[str],
    output_paths: list[Path],
    model_name: str,
    device: str,
    median_depth: float,
    use_ray_pose: bool,
    write_depth: bool,
) -> tuple[dict[str, np.ndarray], float | None]:
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "Depth Anything 3 is required for video depth/pose estimation. "
            'Install with `cd src/vbgs && pip install -e ".[depth]"`.'
        ) from exc

    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=_select_torch_device(device))
    prediction = model.inference(
        [str(path) for path in image_paths],
        use_ray_pose=use_ray_pose,
    )

    depths = _to_numpy(prediction.depth).astype(np.float32)
    poses = {}
    extrinsics = _to_numpy(prediction.extrinsics)
    for stem, extrinsic in zip(stems, extrinsics, strict=True):
        poses[stem] = _w2c_to_opengl_c2w(np.asarray(extrinsic))

    if write_depth:
        for depth, output_path in zip(depths, output_paths, strict=True):
            depth = np.asarray(depth, dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0)
            if valid.any():
                depth = depth / max(float(np.median(depth[valid])), 1e-6)
                depth = depth * float(median_depth)
            np.save(output_path, depth.astype(np.float32))

    camera_angle_x = None
    intrinsics = getattr(prediction, "intrinsics", None)
    if intrinsics is not None and len(intrinsics) > 0:
        first_intrinsic = _to_numpy(intrinsics)[0].astype(np.float64)
        if first_intrinsic.shape[0] >= 3 and first_intrinsic.shape[1] >= 3:
            width = Image.open(image_paths[0]).size[0]
            camera_angle_x = _camera_angle_x_from_intrinsics(
                width,
                first_intrinsic[0, 0],
            )
    return poses, camera_angle_x


def _copy_depths_from_dir(
    depth_dir: Path,
    stems: list[str],
    output_paths: list[Path],
) -> None:
    for stem, output_path in zip(stems, output_paths, strict=True):
        npy = depth_dir / f"{stem}.npy"
        png = depth_dir / f"{stem}.png"
        if npy.exists():
            depth = np.load(npy).astype(np.float32)
        elif png.exists():
            depth = np.asarray(Image.open(png), dtype=np.float32)
        else:
            raise FileNotFoundError(f"Missing depth for {stem}: expected {npy} or {png}")
        np.save(output_path, depth.astype(np.float32))


def preprocess_blender(args: argparse.Namespace) -> None:
    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    total = {"train": 0, "val": 0}
    for split in ("train", "val", "test"):
        transform_path = source / f"transforms_{split}.json"
        if not transform_path.exists():
            continue
        with transform_path.open() as f:
            data = json.load(f)
        frames = []
        for frame in data["frames"]:
            rel = Path(frame["file_path"])
            src_stem = source / rel
            dst_stem = output / rel
            dst_stem.parent.mkdir(parents=True, exist_ok=True)
            for suffix in (".png", "_depth_da3.npy", ".npz"):
                src = Path(f"{src_stem}{suffix}")
                if src.exists():
                    shutil.copy2(src, Path(f"{dst_stem}{suffix}"))
            for depth_png in source.glob(f"{frame['file_path']}_depth_*.png"):
                shutil.copy2(depth_png, output / depth_png.relative_to(source))
            record = dict(frame)
            record["file_path"] = str(rel).replace("\\", "/")
            frames.append(record)
        _write_scene_transforms(
            output,
            split,
            frames,
            data.get("camera_angle_x", DEFAULT_CAMERA_ANGLE_X),
        )
        if split in total:
            total[split] = len(frames)

    _write_scene_manifest(
        output,
        "blender",
        source,
        train_frames=total["train"],
        val_frames=total["val"],
        normalization_ranges=[[-1, 1], [-1, 1], [-1, 1], [0, 1], [0, 1], [0, 1]],
    )
    print(f"wrote standard scene to {output}")


def preprocess_video_scene(args: argparse.Namespace) -> None:
    import cv2

    video = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "_video_frames"
    train_dir = output / "train"
    val_dir = output / "val"
    frames_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")

    source_idx = 0
    written = 0
    extracted = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if source_idx % args.stride != 0:
            source_idx += 1
            continue

        is_val = args.val_stride > 0 and written % args.val_stride == 0
        split = "val" if is_val else "train"
        split_dir = val_dir if is_val else train_dir
        stem = f"r_{written:06d}"
        temp_rgb_path = frames_dir / f"{stem}.png"
        rgb_path = split_dir / f"{stem}.png"
        cv2.imwrite(str(temp_rgb_path), frame)
        shutil.copy2(temp_rgb_path, rgb_path)
        extracted.append(
            {
                "stem": stem,
                "split": split,
                "source_frame": int(source_idx),
                "rgb_path": rgb_path,
                "temp_rgb_path": temp_rgb_path,
                "depth_path": split_dir / f"{stem}_depth_da3.npy",
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
            }
        )

        written += 1
        source_idx += 1
        if args.frames is not None and written >= args.frames:
            break

    cap.release()
    if not extracted:
        raise ValueError(f"No frames extracted from video: {video}")

    da3_poses = {}
    da3_camera_angle_x = None
    needs_da3 = args.depth_source == "da3" or args.pose_source == "da3"
    if needs_da3:
        da3_poses, da3_camera_angle_x = _run_depth_anything3(
            [item["rgb_path"] for item in extracted],
            [item["stem"] for item in extracted],
            [item["depth_path"] for item in extracted],
            args.da3_model,
            args.da3_device,
            args.depth_median,
            args.da3_use_ray_pose,
            args.depth_source == "da3",
        )

    if args.depth_source == "da3":
        pass
    elif args.depth_source == "dir":
        if args.depth_dir is None:
            raise ValueError("--depth-dir is required when --depth-source=dir")
        _copy_depths_from_dir(
            Path(args.depth_dir),
            [item["stem"] for item in extracted],
            [item["depth_path"] for item in extracted],
        )
    elif args.depth_source == "placeholder":
        for item in extracted:
            depth = np.full(
                (item["height"], item["width"]),
                float(args.placeholder_depth),
                dtype=np.float32,
            )
            np.save(item["depth_path"], depth)
    else:
        raise ValueError(f"Unknown depth source: {args.depth_source}")

    if args.pose_source == "da3":
        poses = da3_poses
    elif args.pose_source == "file":
        if args.pose_file is None:
            raise ValueError("--pose-file is required when --pose-source=file")
        poses = _load_pose_json(Path(args.pose_file))
    elif args.pose_source == "stationary":
        poses = {}
    else:
        raise ValueError(f"Unknown pose source: {args.pose_source}")

    train_frames = []
    val_frames = []
    for item in extracted:
        split = item["split"]
        rel_stem = f"{split}/{item['stem']}"
        pose = poses.get(item["stem"], poses.get(rel_stem))
        if pose is None:
            if args.pose_source == "stationary":
                pose = np.asarray(_stationary_blender_pose(), dtype=np.float64)
            else:
                raise KeyError(f"No camera pose found for video frame {item['stem']}")
        record = {
            "file_path": rel_stem,
            "source_frame": item["source_frame"],
            "transform_matrix": np.asarray(pose, dtype=np.float64).tolist(),
        }
        if split == "val":
            val_frames.append(record)
        else:
            train_frames.append(record)

    if not train_frames and val_frames:
        train_frames.append(val_frames.pop(0))

    warning = None
    if args.depth_source == "placeholder" or args.pose_source == "stationary":
        warning = (
            "Video preprocessing used placeholder depth or stationary poses. "
            "Use --depth-source da3 and --pose-source da3 or file "
            "for reconstruction runs."
        )
    camera_angle_x = da3_camera_angle_x or args.camera_angle_x
    _write_scene_transforms(output, "train", train_frames, camera_angle_x)
    _write_scene_transforms(output, "val", val_frames, camera_angle_x)
    _write_scene_transforms(output, "test", val_frames or train_frames, camera_angle_x)
    _write_scene_manifest(
        output,
        "video",
        video,
        train_frames=len(train_frames),
        val_frames=len(val_frames),
        normalization_ranges=[[-1, 1], [-1, 1], [-1, 1], [0, 1], [0, 1], [0, 1]],
        warning=warning,
    )
    print(f"wrote standard video scene to {output}")
    if warning:
        print(warning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    tum = subparsers.add_parser("tum-rgbd")
    tum.add_argument("--input", required=True)
    tum.add_argument("--output", required=True)
    tum.add_argument("--frames", type=int, default=None)
    tum.add_argument("--stride", type=int, default=1)
    tum.add_argument("--val-stride", type=int, default=8)
    tum.set_defaults(func=preprocess_tum)

    blender = subparsers.add_parser("blender")
    blender.add_argument("--input", required=True)
    blender.add_argument("--output", required=True)
    blender.set_defaults(func=preprocess_blender)

    video = subparsers.add_parser("video")
    video.add_argument("--input", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--frames", type=int, default=None)
    video.add_argument("--stride", type=int, default=1)
    video.add_argument("--val-stride", type=int, default=8)
    video.add_argument("--camera-angle-x", type=float, default=DEFAULT_CAMERA_ANGLE_X)
    video.add_argument(
        "--depth-source",
        choices=["da3", "dir", "placeholder"],
        default="da3",
    )
    video.add_argument(
        "--da3-model",
        default="depth-anything/DA3-BASE",
    )
    video.add_argument("--da3-device", default="auto")
    video.add_argument("--da3-use-ray-pose", action="store_true")
    video.add_argument("--depth-median", type=float, default=1.0)
    video.add_argument("--depth-dir", type=Path, default=None)
    video.add_argument(
        "--pose-source",
        choices=["da3", "file", "stationary"],
        default="da3",
    )
    video.add_argument("--pose-file", type=Path, default=None)
    video.add_argument("--placeholder-depth", type=float, default=1.0)
    video.set_defaults(func=preprocess_video_scene)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
