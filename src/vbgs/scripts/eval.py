#!/usr/bin/env python3
"""Evaluate a VBGS Blender model on a transforms split."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import vbgs
from vbgs.io.output import RunOutput
from vbgs.io.paths import add_gaussian_splatting_to_syspath

ROOT = Path(vbgs.__file__).resolve().parent.parent
add_gaussian_splatting_to_syspath()

from scene.dataset_readers import CameraInfo  # noqa: E402
from utils.graphics_utils import focal2fov  # noqa: E402
from vbgs.render.volume import render_img, vbgs_model_to_splat  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../../data/blender/lego"),
    )
    parser.add_argument("--split", default="transforms_val.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save predicted renders and GT-vs-pred comparison images.",
    )
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--compare-dir", type=Path)
    return parser.parse_args()


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


def psnr(pred, gt):
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def main():
    args = parse_args()
    model_path = args.model.resolve()
    data_path = (ROOT / args.data_path).resolve()
    output = RunOutput.create(output_dir=model_path.parent)
    out_path = args.out or model_path.parent / "val_psnr.json"
    render_dir = args.render_dir or output.path / "renders_val"
    compare_dir = args.compare_dir or output.path / "renders_val_compare"
    if args.save_images:
        render_dir = output.image_dir(render_dir.name) if args.render_dir is None else render_dir
        compare_dir = output.image_dir(compare_dir.name) if args.compare_dir is None else compare_dir
        output.ensure_dir(render_dir)
        output.ensure_dir(compare_dir)

    with open(data_path / args.split) as f:
        meta = json.load(f)
    frames = meta["frames"]
    angle_x = float(meta["camera_angle_x"])

    model = vbgs_model_to_splat(model_path)
    values = []
    for idx, frame in enumerate(frames):
        camera = make_camera(idx, frame, data_path, angle_x)
        pred = np.clip(render_img(model, [camera], 0, bg=0), 0.0, 1.0)
        gt = np.asarray(camera.image, dtype=np.float32) / 255.0
        values.append(psnr(pred, gt))
        if args.save_images:
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            stem = f"{idx:03d}_{Path(frame['file_path']).name}"
            pred_img.save(render_dir / f"{stem}.png")

            width, height = camera.image.size
            canvas = Image.new("RGB", (2 * width, height + 28), "white")
            canvas.paste(camera.image, (0, 28))
            canvas.paste(pred_img, (width, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), "GT", fill=(0, 0, 0))
            draw.text((width + 8, 6), "Prediction", fill=(0, 0, 0))
            canvas.save(compare_dir / f"{stem}_gt_vs_pred.png")
        if (idx + 1) % 10 == 0:
            print(f"{idx + 1}/{len(frames)} mean={np.mean(values):.4f}")

    result = {
        "split": args.split,
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
