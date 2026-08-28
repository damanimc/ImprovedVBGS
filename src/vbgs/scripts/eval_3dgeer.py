#!/usr/bin/env python3
"""Evaluate a VBGS model with Bosch 3DGEER (geer-rasterizer, pinhole / PH mode).

Requires ``bash install_3dgeer.sh`` first. The geer CUDA API is not a drop-in for
``eval.py`` (graphdeco); this script builds PH-mode settings and calls geer
directly. On NeRF Synthetic Lego we measured ~+0.28 dB mean val PSNR vs
graphdeco on the same model.
"""

from __future__ import annotations

import argparse
import json
import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import vbgs
from vbgs.io.output import RunOutput
from vbgs.io.paths import add_gaussian_splatting_to_syspath

ROOT = Path(vbgs.__file__).resolve().parent.parent
add_gaussian_splatting_to_syspath()

from arguments import PipelineParams  # noqa: E402
from scene.dataset_readers import CameraInfo  # noqa: E402
from utils.camera_utils import loadCam  # noqa: E402
from utils.graphics_utils import focal2fov  # noqa: E402
from diff_gaussian_rasterization import (  # noqa: E402
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from vbgs.render.volume import vbgs_model_to_splat  # noqa: E402

SAMPLE_STEP = 0.002  # 3DGEER default


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VBGS with 3DGEER geer-rasterizer (PH mode)"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../../data/blender/lego"),
    )
    parser.add_argument("--split", default="transforms_val.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument(
        "--sample-step",
        type=float,
        default=SAMPLE_STEP,
        help="3DGEER ray-grid spacing (radians); default matches 3DGEER train.py",
    )
    return parser.parse_args()


def psnr(pred, gt):
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def fov_sample2ray(fovx, fovy, interval):
    theta_arr = torch.arange(interval / 2, fovx, interval)
    theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
    phi_arr = torch.arange(interval / 2, fovy, interval)
    phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))
    return theta_arr.float(), phi_arr.float()


def mirror_transform(m, z, xi=0.0):
    return m / (1 + xi * (z / (torch.abs(z))) * (1 + m**2) ** 0.5)


def attach_geer_ph(cam, sample_step=SAMPLE_STEP):
    """Attach 3DGEER PH-mode fields onto a graphdeco Camera from loadCam."""
    w = int(cam.image_width)
    h = int(cam.image_height)
    fx = w / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = h / (2.0 * math.tan(cam.FoVy * 0.5))
    cam.focal_x = fx
    cam.focal_y = fy
    cam.principal_x = w / 2.0
    cam.principal_y = h / 2.0
    cam.render_model = 2  # PH
    cam.distortion_coeffs = None
    cam.raymap = None

    arr_theta, arr_phi = fov_sample2ray(cam.FoVx / 2, cam.FoVy / 2, sample_step)
    cos_theta = torch.cos(arr_theta)
    cos_phi = torch.cos(arr_phi)
    cos_theta = torch.where(
        torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta
    )
    cos_phi = torch.where(
        torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi
    )
    device = cam.world_view_transform.device
    cam.tan_theta = torch.tan(arr_theta).to(device)
    cam.tan_phi = torch.tan(arr_phi).to(device)
    cam.mirror_transformed_tan_theta = mirror_transform(
        cam.tan_theta, cos_theta.to(device)
    ).float()
    cam.mirror_transformed_tan_phi = mirror_transform(
        cam.tan_phi, cos_phi.to(device)
    ).float()
    return cam


def render_geer(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    antialiasing = bool(getattr(pipe, "antialiasing", False))

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        mirror_transformed_tan_theta=viewpoint_camera.mirror_transformed_tan_theta,
        mirror_transformed_tan_phi=viewpoint_camera.mirror_transformed_tan_phi,
        tan_theta=viewpoint_camera.tan_theta,
        tan_phi=viewpoint_camera.tan_phi,
        focal_x=float(viewpoint_camera.focal_x),
        focal_y=float(viewpoint_camera.focal_y),
        principal_x=float(viewpoint_camera.principal_x),
        principal_y=float(viewpoint_camera.principal_y),
        distortion_coeffs=torch.empty(0, device="cuda"),
        raymap=torch.empty(0, device="cuda"),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=bool(getattr(pipe, "debug", False)),
        antialiasing=antialiasing,
        render_mode=int(viewpoint_camera.render_model),
        near_threshold=0.2,
        asso_mode=0,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, *_rest = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
    )
    rendered_image = rendered_image.clamp(0, 1)
    return rendered_image.detach().cpu().permute(1, 2, 0).numpy()


def make_camera(idx, frame, data_path, angle_x):
    image_path = data_path / f"{frame['file_path']}.png"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    fovy = focal2fov(0.5 * width / math.tan(0.5 * angle_x), height)
    return CameraInfo(
        uid=idx,
        R=w2c[:3, :3].T,
        T=w2c[:3, 3],
        FovY=fovy,
        FovX=angle_x,
        image=image,
        image_path=str(image_path),
        image_name=image_path.stem,
        width=width,
        height=height,
    )


def _require_geer_rasterizer():
    fields = set(GaussianRasterizationSettings.__annotations__)
    need = {"tan_theta", "render_mode", "focal_x"}
    if not need.issubset(fields):
        raise RuntimeError(
            "diff_gaussian_rasterization is not the 3DGEER geer-rasterizer. "
            "From src/vbgs run: bash install_3dgeer.sh"
        )


def main():
    args = parse_args()
    _require_geer_rasterizer()

    model_path = args.model.resolve()
    data_path = (ROOT / args.data_path).resolve()
    output = RunOutput.create(output_dir=model_path.parent)
    out_path = args.out or model_path.parent / "val_psnr_3dgeer.json"
    render_dir = args.render_dir or output.path / "renders_val_3dgeer"
    compare_dir = args.compare_dir or output.path / "renders_val_3dgeer_compare"
    if args.save_images:
        render_dir = (
            output.image_dir(render_dir.name)
            if args.render_dir is None
            else render_dir
        )
        compare_dir = (
            output.image_dir(compare_dir.name)
            if args.compare_dir is None
            else compare_dir
        )
        output.ensure_dir(render_dir)
        output.ensure_dir(compare_dir)

    pipe_parser = ArgumentParser()
    pipe = PipelineParams(pipe_parser)
    if not hasattr(pipe, "antialiasing"):
        pipe.antialiasing = False

    class CArgs:
        resolution = -1
        data_device = "cuda:0"

    cargs = CArgs()
    model = vbgs_model_to_splat(model_path)

    with open(data_path / args.split) as f:
        meta = json.load(f)
    frames = meta["frames"]
    angle_x = float(meta["camera_angle_x"])
    bg = torch.zeros(3, device="cuda")

    values = []
    for idx, frame in enumerate(frames):
        cam_info = make_camera(idx, frame, data_path, angle_x)
        cam = loadCam(cargs, id=0, cam_info=cam_info, resolution_scale=1.0)
        cam = attach_geer_ph(cam, sample_step=args.sample_step)
        pred = np.clip(render_geer(cam, model, pipe, bg), 0.0, 1.0)
        gt = np.asarray(cam_info.image, dtype=np.float32) / 255.0
        if gt.shape[:2] != pred.shape[:2]:
            gt = (
                np.asarray(
                    Image.fromarray((gt * 255).astype(np.uint8)).resize(
                        (pred.shape[1], pred.shape[0]), Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
        values.append(psnr(pred, gt))
        if args.save_images:
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            stem = f"{idx:03d}_{Path(frame['file_path']).name}"
            pred_img.save(render_dir / f"{stem}.png")
            width, height = pred_img.size
            canvas = Image.new("RGB", (2 * width, height + 28), "white")
            gt_img = Image.fromarray((gt * 255).astype(np.uint8))
            canvas.paste(gt_img, (0, 28))
            canvas.paste(pred_img, (width, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), "GT", fill=(0, 0, 0))
            draw.text((width + 8, 6), "3DGEER", fill=(0, 0, 0))
            canvas.save(compare_dir / f"{stem}_gt_vs_pred.png")
        if (idx + 1) % 10 == 0:
            print(f"{idx + 1}/{len(frames)} mean={np.mean(values):.4f}")

    result = {
        "split": args.split,
        "rasterizer": "3dgeer-geer-rasterizer-PH",
        "n": len(values),
        "mean_psnr": float(np.mean(values)),
        "std_psnr": float(np.std(values)),
        "min_psnr": float(np.min(values)),
        "max_psnr": float(np.max(values)),
        "values": values,
    }
    if args.out is None:
        output.metrics(result, out_path.name)
    else:
        output.ensure_dir(out_path.parent)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["n", "mean_psnr", "std_psnr", "min_psnr", "max_psnr"]
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    if args.save_images:
        print(f"renders: {render_dir}")
        print(f"comparisons: {compare_dir}")


if __name__ == "__main__":
    main()
