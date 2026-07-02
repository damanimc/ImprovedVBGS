import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.random as jr
import numpy as np
from tqdm import tqdm

PACKAGE_PATH = Path(__file__).resolve().parents[1]
if str(PACKAGE_PATH) not in sys.path:
    sys.path.append(str(PACKAGE_PATH))

from model_volume import get_volume_delta_mixture
from vbgs.data.scene import SceneDataIterator
from vbgs.data.utils import create_normalizing_params, normalize_data
from vbgs.io.output import RunOutput, default_output_root
from vbgs.model.continual import (
    build_sparse_index,
    default_densify_stats,
    initialize_mean,
    query_candidate_indices,
    reset_component_stats,
    strict_unused_count,
)
from vbgs.model.densify import densify_from_frame
from vbgs.model.eval import eval_elbo
from vbgs.model.reassign import reassign
from vbgs.model.train import fit_gmm_step


def parse_args():
    parser = argparse.ArgumentParser(description="Train VBGS on a standard RGB-D scene")
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="scene")
    parser.add_argument("--components", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=["auto", "train", "test", "val"],
        default="auto",
    )
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-batch-size", type=int, default=50_000)
    parser.add_argument("--eval-subsample", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--semantic-classes", type=int, default=0)
    parser.add_argument("--precision", choices=["fp64", "fp32", "tf32"], default="fp64")
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--candidate-m", type=int, default=128)
    parser.add_argument("--candidate-eps", type=float, default=0.0)
    parser.add_argument("--init", choices=["random", "first-frame", "full-data"], default="random")
    parser.add_argument("--init-downsample-factor", type=int, default=None)
    parser.add_argument("--densify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--densify-point-ratio", type=float, default=0.0)
    parser.add_argument("--densify-unseen-distance-threshold", type=float, default=None)
    parser.add_argument("--densify-min-unseen-fraction", type=float, default=0.0)
    parser.add_argument("--densify-min-unseen-points", type=int, default=0)
    parser.add_argument("--densify-reassign-if-full", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reassign", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reassign-every", type=int, default=1)
    parser.add_argument("--reassign-fraction", type=float, default=0.05)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Record per-frame timings and write metrics.json",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load all training frames into RAM before the training loop",
    )
    return parser.parse_args()


def load_scene_manifest(data_path):
    manifest_path = data_path / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as f:
        return json.load(f)


def scene_data_params(manifest):
    ranges = manifest.get("normalization_ranges")
    if ranges is None:
        ranges = [[-1, 1], [-1, 1], [-1, 1], [0, 1], [0, 1], [0, 1]]
    return create_normalizing_params(*ranges)


def has_split(data_path, split):
    path = data_path / f"transforms_{split}.json"
    if not path.exists():
        return False
    with path.open() as f:
        data = json.load(f)
    return len(data.get("frames", [])) > 0


def resolve_split(data_path, split):
    if split != "auto":
        return split
    for candidate in ("train", "test", "val"):
        if has_split(data_path, candidate) and len(
            SceneDataIterator(data_path, split=candidate)
        ):
            return candidate
    raise ValueError("Need at least 1 loadable frame")


def make_data_iter(data_path, split, data_params, subsample, preload=False):
    data_iter = SceneDataIterator(
        data_path,
        split=split,
        data_params=data_params,
        subsample=subsample,
    )
    if len(data_iter) < 1:
        raise ValueError("Need at least 1 loadable frame")
    return data_iter


def prepare_data_iter(data_iter, frames, preload):
    if frames is not None:
        data_iter._frames = data_iter._frames[:frames]
    if preload:
        data_iter.preload_frames()
    return data_iter


def load_frame_and_candidates(
    data_iter, step, candidate_tree, candidate_m, n_components, candidate_eps
):
    x = data_iter._get_frame(step)
    if candidate_tree is None:
        return x, None
    return x, query_candidate_indices(
        candidate_tree,
        x[:, :3],
        candidate_m,
        n_components,
        eps=candidate_eps,
    )


def frame_record(step, x, *, densify_stats, **timings):
    return {
        "frame": int(step),
        "points": int(x.shape[0]),
        **timings,
        "available_components": int(densify_stats["available_components"]),
        "strict_unused_components": int(densify_stats["strict_unused_components"]),
        "component_offset": int(densify_stats["component_offset"]),
        "densify_points": int(densify_stats["densify_points"]),
        "densified_components": int(densify_stats["densified_components"]),
        "densify_distance_threshold": densify_stats["densify_distance_threshold"],
        "densify_distance_mean": densify_stats["densify_distance_mean"],
        "unseen_distance_threshold": densify_stats["unseen_distance_threshold"],
        "unseen_points": int(densify_stats["unseen_points"]),
        "unseen_fraction": float(densify_stats["unseen_fraction"]),
        "densify_skipped_reason": densify_stats["densify_skipped_reason"],
        "recycle_mode": bool(densify_stats["recycle_mode"]),
        "densify_reassign_fraction": densify_stats["reassign_fraction"],
    }


def main():
    args = parse_args()
    jax.config.update("jax_default_device", jax.devices()[args.device])
    output = RunOutput.create(
        run_name=args.run_name,
        output_root=args.output_root or default_output_root(),
        output_dir=args.output_dir,
        unique=args.output_dir is None,
    )

    key = jr.PRNGKey(args.seed)
    manifest = load_scene_manifest(args.data_path)
    data_params = scene_data_params(manifest)
    train_split = resolve_split(args.data_path, args.split)
    data_iter = prepare_data_iter(
        make_data_iter(
            args.data_path, train_split, data_params, args.subsample
        ),
        args.frames,
        args.preload,
    )

    init_random = args.init == "random"
    init_first_frame = args.init == "first-frame"
    x_data = None
    if init_first_frame:
        x_data = data_iter._get_frame(0)
        if args.init_downsample_factor is not None and args.init_downsample_factor > 1:
            x_data = x_data[:: int(args.init_downsample_factor)]
    elif args.init == "full-data":
        chunks = [x for x in data_iter]
        data = np.concatenate(chunks, axis=0)
        data, data_params = normalize_data(data)
        rng = np.random.default_rng(args.seed)
        idcs = rng.permutation(data.shape[0])[: args.components]
        x_data = data[idcs]
        data_iter = prepare_data_iter(
            make_data_iter(
                args.data_path, train_split, data_params, args.subsample
            ),
            args.frames,
            args.preload,
        )

    feature_dim = 6 + max(0, int(args.semantic_classes))
    if x_data is not None and x_data.shape[1] != feature_dim:
        raise ValueError(
            f"Expected {feature_dim} features for semantic_classes={args.semantic_classes}, "
            f"got {x_data.shape[1]}"
        )

    key, subkey = jr.split(key)
    mean_init, init_metrics = initialize_mean(
        key=subkey,
        init_data=x_data,
        n_components=args.components,
        event_shape=(feature_dim, 1),
        init_random=init_random,
        init_first_frame=init_first_frame,
        add_noise=not init_first_frame,
    )
    key, subkey = jr.split(key)
    prior_model = get_volume_delta_mixture(
        key=subkey,
        n_components=args.components,
        mean_init=mean_init,
        beta=0,
        learning_rate=1,
        dof_offset=1,
        position_scale=args.components,
        position_event_shape=(3, 1),
        semantic_event_shape=(
            (int(args.semantic_classes), 1)
            if int(args.semantic_classes) > 0
            else None
        ),
    )
    model = copy.deepcopy(prior_model)

    candidate_tree, topm_cache = build_sparse_index(
        prior_model,
        top_m=args.top_m,
        candidate_m=args.candidate_m,
        precision=args.precision,
    )
    prior_stats, space_stats, color_stats = None, None, None
    metrics = (
        {
            "data": str(args.data_path),
            "source_type": manifest.get("source_type"),
            "components": args.components,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "top_m": args.top_m,
            "candidate_m": args.candidate_m,
            "candidate_eps": args.candidate_eps,
            "semantic_classes": int(args.semantic_classes),
            "init": args.init,
            **init_metrics,
            "densify": args.densify,
            "densify_point_ratio": args.densify_point_ratio,
            "densify_unseen_distance_threshold": args.densify_unseen_distance_threshold,
            "densify_min_unseen_fraction": args.densify_min_unseen_fraction,
            "densify_min_unseen_points": args.densify_min_unseen_points,
            "densify_reassign_if_full": args.densify_reassign_if_full,
            "reassign": args.reassign,
            "reassign_every": args.reassign_every,
            "reassign_fraction": args.reassign_fraction,
            "preload": args.preload,
            "split": train_split,
            "train_frames": len(data_iter),
            "eval_enabled": args.eval,
            "frames": [],
            "eval": [],
        }
        if args.debug
        else {}
    )

    n_frames = len(data_iter)
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    prefetch = None

    for step in tqdm(range(n_frames), total=n_frames, desc="training", unit="frame"):
        if prefetch is not None:
            x, candidate_indices = prefetch.result()
        else:
            x, candidate_indices = load_frame_and_candidates(
                data_iter,
                step,
                candidate_tree,
                args.candidate_m,
                args.components,
                args.candidate_eps,
            )
        if args.debug:
            start = time.perf_counter()
            densify_seconds = 0.0
            reassign_seconds = 0.0
            sparse_rebuild_seconds = 0.0
            kdtree_seconds = 0.0
        densify_stats = default_densify_stats(args.densify_unseen_distance_threshold)
        if args.densify and not (init_first_frame and step == 0):
            if args.debug:
                densify_start = time.perf_counter()
            available = strict_unused_count(model, prior_model)
            densify_stats["available_components"] = available
            densify_stats["strict_unused_components"] = available
            if available > 0 or args.densify_reassign_if_full:
                densify_candidates = query_candidate_indices(
                    candidate_tree,
                    x[:, :3],
                    args.candidate_m,
                    args.components,
                )
                prior_model, densify_stats, inserted_components = densify_from_frame(
                    prior_model,
                    model,
                    x,
                    candidate_indices=densify_candidates,
                    point_ratio=(
                        args.densify_point_ratio if args.densify_point_ratio > 0 else None
                    ),
                    unseen_distance_threshold=args.densify_unseen_distance_threshold,
                    min_unseen_fraction=args.densify_min_unseen_fraction,
                    min_unseen_points=args.densify_min_unseen_points,
                    recycle_if_full=args.densify_reassign_if_full,
                    reassign_fraction=args.reassign_fraction,
                    debug=True,
                    return_indices=True,
                )
                prior_stats = reset_component_stats(prior_stats, inserted_components)
                space_stats = reset_component_stats(space_stats, inserted_components)
                color_stats = reset_component_stats(color_stats, inserted_components)
            if args.debug:
                densify_seconds = time.perf_counter() - densify_start
            if densify_stats["densified_components"] > 0:
                if args.debug:
                    rebuild_start = time.perf_counter()
                candidate_tree, topm_cache = build_sparse_index(
                    prior_model,
                    top_m=args.top_m,
                    candidate_m=args.candidate_m,
                    precision=args.precision,
                )
                if args.debug:
                    sparse_rebuild_seconds += time.perf_counter() - rebuild_start

        if args.debug:
            fit_start = time.perf_counter()
        low_elbo_count = (
            max(1024, int(args.components * args.reassign_fraction * 4))
            if (
                args.reassign
                and args.reassign_every > 0
                and step % args.reassign_every == 0
            )
            else 0
        )
        fit_result = fit_gmm_step(
            prior_model,
            model,
            data=x,
            batch_size=args.batch_size,
            prior_stats=prior_stats,
            space_stats=space_stats,
            color_stats=color_stats,
            precision=args.precision,
            top_m=args.top_m,
            candidate_indices=candidate_indices,
            topm_cache=topm_cache,
            low_elbo_count=low_elbo_count,
        )
        if low_elbo_count:
            (
                model,
                prior_stats,
                space_stats,
                color_stats,
                _semantic_stats,
                low_elbo,
            ) = fit_result
        else:
            model, prior_stats, space_stats, color_stats, _semantic_stats = fit_result
        if low_elbo_count:
            if args.debug:
                reassign_start = time.perf_counter()
            prior_model = reassign(
                prior_model,
                model,
                x,
                args.batch_size,
                args.reassign_fraction,
                precision=args.precision,
                point_indices=low_elbo["point_indices"],
                point_elbos=low_elbo["elbo"],
            )
            if args.debug:
                reassign_seconds = time.perf_counter() - reassign_start
                rebuild_start = time.perf_counter()
            candidate_tree, topm_cache = build_sparse_index(
                prior_model,
                top_m=args.top_m,
                candidate_m=args.candidate_m,
                precision=args.precision,
            )
            if args.debug:
                sparse_rebuild_seconds += time.perf_counter() - rebuild_start
        jax.block_until_ready(model.mixture.likelihood.mean)
        if step + 1 < n_frames:
            prefetch = prefetch_executor.submit(
                load_frame_and_candidates,
                data_iter,
                step + 1,
                candidate_tree,
                args.candidate_m,
                args.components,
                args.candidate_eps,
            )
        else:
            prefetch = None
        if args.debug:
            metrics["frames"].append(
                frame_record(
                    step,
                    x,
                    seconds=time.perf_counter() - start,
                    fit_seconds=time.perf_counter() - fit_start,
                    densify_seconds=densify_seconds,
                    reassign_seconds=reassign_seconds,
                    kdtree_seconds=kdtree_seconds,
                    sparse_rebuild_seconds=sparse_rebuild_seconds,
                    densify_stats=densify_stats,
                )
            )
        if args.save_every is not None and (step + 1) % args.save_every == 0:
            output.checkpoint(model, data_params, f"model_{step:03d}.json")

    prefetch_executor.shutdown(wait=False)

    if args.eval and has_split(args.data_path, "val"):
        eval_iter = SceneDataIterator(
            args.data_path,
            split="val",
            data_params=data_params,
            subsample=None,
        )
        eval_frames = [x for x in eval_iter]
        eval_results = eval_elbo(
            model,
            eval_frames,
            args.eval_batch_size,
            args.precision,
            args.eval_subsample,
            args.seed,
            top_m=args.top_m,
            candidate_m=args.candidate_m,
        )
        if args.debug:
            metrics["eval"] = eval_results

    paths = output.final_model(model, data_params, metrics)
    if args.debug:
        output.metrics(metrics)
    print(f"done: {paths['final_model']}")


if __name__ == "__main__":
    main()
