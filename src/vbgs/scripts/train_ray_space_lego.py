#!/usr/bin/env python3
"""Train Ray-space VBGS on NeRF Synthetic Lego and report val PSNR.

Pipeline (CPU):
1. Load Blender rays + RGB (+ depth prior when available).
2. Fit colored RVBGS (latent or fixed depth).
3. Export a renderable Gaussian model:
   - mixture component means, plus
   - densified residual points from refined ray hits.
4. CPU EWA alpha renderer → val PSNR (black background, NeRF-Synthetic).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_PATH = Path(__file__).resolve().parents[1]
if str(PACKAGE_PATH) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PATH))

from vbgs.ray_space.blender_data import load_ray_dataset
from vbgs.ray_space.colored import fit_colored_ray_vbgs, state_to_model_dict
from vbgs.ray_space.geometry import point_on_ray
from vbgs.ray_space.render_cpu import eval_split_psnr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=Path, default=Path("/workspace/data/blender/lego"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/src/vbgs/output/lego_ray_space"),
    )
    p.add_argument("--train-split", default="test")
    p.add_argument("--eval-split", default="val")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--eval-frames", type=int, default=20)
    p.add_argument("--components", type=int, default=256)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--image-scale", type=float, default=0.25)
    p.add_argument("--max-pixels-per-frame", type=int, default=8000)
    p.add_argument("--depth-prior-precision", type=float, default=100.0)
    p.add_argument("--splat-points", type=int, default=8000)
    p.add_argument("--fix-depth", action="store_true")
    p.add_argument(
        "--noisy-depth-std",
        type=float,
        default=0.0,
        help="If >0, corrupt depth prior then let RVBGS refine (ray-space demo).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def densify_model_with_points(
    model: dict,
    pts: np.ndarray,
    rgb: np.ndarray,
    n_points: int,
    seed: int = 0,
    cov: float = 0.006,
) -> dict:
    """Merge mixture means with a subsample of refined ray-hit points."""
    rng = np.random.default_rng(seed)
    mu_m = np.asarray(model["mu"], dtype=np.float64)
    si_m = np.asarray(model["si"], dtype=np.float64)
    a_m = np.asarray(model["alpha"], dtype=np.float64)
    k_m = len(a_m)

    n_points = min(n_points, len(pts))
    idx = rng.choice(len(pts), size=n_points, replace=False)
    mu_p = np.concatenate([pts[idx], rgb[idx]], axis=1)
    si_p = np.zeros((n_points, 6, 6), dtype=np.float64)
    for i in range(n_points):
        si_p[i, :3, :3] = cov * np.eye(3)
        si_p[i, 3:, 3:] = 0.01 * np.eye(3)

    # Mixture gets 30% mass, densified points 70% (coverage).
    a_m = 0.3 * a_m / max(a_m.sum(), 1e-12)
    a_p = np.full(n_points, 0.7 / n_points)
    mu = np.concatenate([mu_m, mu_p], axis=0)
    si = np.concatenate([si_m, si_p], axis=0)
    alpha = np.concatenate([a_m, a_p], axis=0)
    return {
        "mu": mu.tolist(),
        "si": si.tolist(),
        "alpha": alpha.tolist(),
        "n_semantic": 0,
        "n_mixture": int(k_m),
        "n_points": int(n_points),
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Lego rays…", flush=True)
    t0 = time.time()
    data = load_ray_dataset(
        args.data_path,
        split=args.train_split,
        n_frames=args.frames,
        image_scale=args.image_scale,
        max_pixels_per_frame=args.max_pixels_per_frame,
        seed=args.seed,
        prefer_depth_split_fallback=True,
    )
    depth_prior = data["depth_prior"].copy()
    if args.noisy_depth_std > 0:
        rng = np.random.default_rng(args.seed)
        depth_prior = np.maximum(
            depth_prior + rng.normal(0.0, args.noisy_depth_std, size=depth_prior.shape),
            0.2,
        )
        print(
            f"  corrupted depth prior std={args.noisy_depth_std} "
            f"med_err={np.median(np.abs(depth_prior - data['depth_prior'])):.3f}",
            flush=True,
        )
    print(
        f"  split={data['split']} frames={data['n_frames']} rays={len(data['C'])} "
        f"load_s={time.time()-t0:.1f}",
        flush=True,
    )

    print("Fitting colored Ray-space VBGS…", flush=True)
    t1 = time.time()
    state = fit_colored_ray_vbgs(
        data["C"],
        data["d"],
        data["rgb"],
        n_components=args.components,
        n_iters=args.iters,
        depth_prior=depth_prior,
        depth_prior_precision=args.depth_prior_precision,
        fix_depth=args.fix_depth,
        seed=args.seed,
        temperature=1.5,
    )
    print(f"  fit_s={time.time()-t1:.1f}", flush=True)

    pts = point_on_ray(data["C"], data["d"], state.depth_mean)
    depth_err = float(np.median(np.abs(state.depth_mean - data["depth_prior"])))
    print(f"  median |Δλ| vs loaded prior={depth_err:.4f}", flush=True)

    mix = state_to_model_dict(state)
    model = densify_model_with_points(
        mix, pts, data["rgb"], n_points=args.splat_points, seed=args.seed
    )
    model_path = out / "model_final.json"
    with open(model_path, "w") as f:
        json.dump(model, f)
    print(
        f"Wrote {model_path}  mixture={model['n_mixture']} points={model['n_points']}",
        flush=True,
    )

    print(f"Evaluating PSNR on {args.eval_split}…", flush=True)
    t2 = time.time()
    metrics = eval_split_psnr(
        model_path,
        args.data_path,
        split=args.eval_split,
        n_frames=args.eval_frames,
        image_scale=args.image_scale,
        save_dir=out / "renders_val_compare",
    )
    metrics.update(
        {
            "train_split": data["split"],
            "n_train_frames": data["n_frames"],
            "n_rays": int(len(data["C"])),
            "n_components_requested": args.components,
            "n_mixture": model["n_mixture"],
            "n_points": model["n_points"],
            "iters": args.iters,
            "fix_depth": bool(args.fix_depth),
            "noisy_depth_std": args.noisy_depth_std,
            "depth_prior_precision": args.depth_prior_precision,
            "median_depth_delta": depth_err,
            "fit_seconds": time.time() - t1,
            "eval_seconds": time.time() - t2,
            "total_seconds": time.time() - t0,
            "background": 0.0,
        }
    )
    metrics_path = out / "val_psnr.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"\nMean PSNR={metrics['mean_psnr']:.3f} dB  "
        f"(n={metrics['n']}, scale={args.image_scale}) → {metrics_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
