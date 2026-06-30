import argparse
import copy
import json
import sys
import time
from pathlib import Path

import jax
import jax.random as jr
import numpy as np
from tqdm import tqdm

PACKAGE_PATH = Path(__file__).resolve().parents[1]
if str(PACKAGE_PATH) not in sys.path:
    sys.path.append(str(PACKAGE_PATH))

from model_volume import get_volume_delta_mixture
from vbgs.data.habitat import TUMDataIterator
from vbgs.data.utils import create_normalizing_params
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
from vbgs.model.eval import eval_elbo, sparse_elbo_fn
from vbgs.model.precision import PrecisionMap
from vbgs.model.precision_training import ensure_op_precision_runtime
from vbgs.model.reassign import reassign
from vbgs.model.train import fit_gmm_step

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="tum")
    parser.add_argument("--components", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=2 / 3)
    parser.add_argument("--semantic-classes", type=int, default=0)
    parser.add_argument("--precision", choices=["fp64", "fp32", "tf32", "op", "auto"], default="op")
    parser.add_argument("--precision-tolerance", type=float, default=1e-6)
    parser.add_argument("--precision-search-frame", type=int, default=1)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--candidate-m", type=int, default=128)
    parser.add_argument("--init-first-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--densify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--densify-point-ratio", type=float, default=0.0)
    parser.add_argument("--densify-unseen-distance-threshold", type=float, default=None)
    parser.add_argument("--densify-min-unseen-fraction", type=float, default=0.0)
    parser.add_argument("--densify-min-unseen-points", type=int, default=0)
    parser.add_argument("--densify-reassign-if-full", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reassign", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reassign-every", type=int, default=1)
    parser.add_argument("--reassign-fraction", type=float, default=0.05)
    parser.add_argument("--eval-batch-size", type=int, default=50_000)
    parser.add_argument("--eval-subsample", type=int, default=None)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=None)
    return parser.parse_args()


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
    data_params = create_normalizing_params(
        [-5, 5], [-5, 5], [-5, 5], [0, 1], [0, 1], [0, 1]
    )
    data_iter = TUMDataIterator(args.data_path, data_params)
    if args.frames is not None:
        data_iter._frames = data_iter._frames[: args.frames]
    n_total = len(data_iter)
    if n_total < 1:
        raise ValueError("Need at least 1 TUM frame")
    if args.eval:
        if n_total < 2:
            raise ValueError("Need at least 2 TUM frames for train/eval split")
        n_train = int(np.ceil(n_total * args.train_fraction))
        n_train = min(max(n_train, 1), n_total - 1)
        train_frame_paths = data_iter._frames[:n_train]
        eval_frame_paths = data_iter._frames[n_train:]
        data_iter._frames = train_frame_paths
        eval_iter = TUMDataIterator(args.data_path, data_params)
        eval_iter._frames = eval_frame_paths
    else:
        eval_iter = None

    key, subkey = jr.split(key)
    init_data = data_iter._get_frame(0) if args.init_first_frame else None
    feature_dim = 6 + max(0, int(args.semantic_classes))
    if init_data is not None and init_data.shape[1] != feature_dim:
        raise ValueError(
            f"Expected {feature_dim} features for semantic_classes="
            f"{args.semantic_classes}, got {init_data.shape[1]}"
        )
    mean_init, init_metrics = initialize_mean(
        key=subkey,
        init_data=init_data,
        n_components=args.components,
        event_shape=(feature_dim, 1),
        init_random=not args.init_first_frame,
        init_first_frame=args.init_first_frame,
        add_noise=not args.init_first_frame,
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

    sparse_e_step = args.top_m is not None and args.candidate_m is not None
    precision = "fp64" if args.precision == "auto" else args.precision
    precision_search_enabled = args.precision in ("op", "auto") and not (
        sparse_e_step and args.reassign_every <= 0
    )
    if args.precision == "op" and not precision_search_enabled:
        # Sparse top-M path does not use dense ELBO op maps; skip expensive search.
        precision = "fp64"
    precision_map = PrecisionMap(tolerance=args.precision_tolerance)
    prior_stats, space_stats, color_stats = None, None, None

    candidate_tree, topm_cache = build_sparse_index(
        prior_model,
        top_m=args.top_m,
        candidate_m=args.candidate_m,
        precision=precision,
    )

    metrics = {
        "data": str(args.data_path),
        "components": args.components,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "effective_precision": precision,
        "top_m": args.top_m,
        "candidate_m": args.candidate_m,
        "semantic_classes": int(args.semantic_classes),
        "init_first_frame": args.init_first_frame,
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
        "train_fraction": args.train_fraction,
        "eval_enabled": args.eval,
        "eval_batch_size": args.eval_batch_size,
        "eval_subsample": args.eval_subsample,
        "eval_sparse": args.top_m is not None and args.candidate_m is not None,
        "train_frames": len(data_iter),
        "eval_frames": len(eval_iter) if eval_iter is not None else 0,
        "frames": [],
        "eval": [],
    }
    for step, x in tqdm(enumerate(data_iter), total=len(data_iter)):
        start = time.perf_counter()
        densify_seconds = 0.0
        densify_stats = default_densify_stats(args.densify_unseen_distance_threshold)
        reassign_seconds = 0.0
        sparse_rebuild_seconds = 0.0
        if precision_search_enabled and (
            args.precision == "op"
            or (args.precision == "auto" and step >= args.precision_search_frame)
        ):
            precision, precision_map = ensure_op_precision_runtime(
                mode=args.precision,
                frame_idx=step,
                search_frame=args.precision_search_frame,
                prior_model=prior_model,
                normalized=x,
                batch_size=args.batch_size,
                output_dir=output.path,
                tolerance=args.precision_tolerance,
                white_noise_key=jr.PRNGKey(args.seed + 1),
            )
        if args.densify and not (args.init_first_frame and step == 0):
            densify_start = time.perf_counter()
            available = strict_unused_count(model, prior_model)
            densify_stats["available_components"] = available
            densify_stats["strict_unused_components"] = available
            densify_stats["component_offset"] = 0
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
                        args.densify_point_ratio
                        if args.densify_point_ratio > 0
                        else None
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
            densify_seconds = time.perf_counter() - densify_start
            if densify_stats["densified_components"] > 0:
                rebuild_start = time.perf_counter()
                candidate_tree, topm_cache = build_sparse_index(
                    prior_model,
                    top_m=args.top_m,
                    candidate_m=args.candidate_m,
                    precision=precision,
                )
                sparse_rebuild_seconds += time.perf_counter() - rebuild_start
        if args.reassign and args.reassign_every > 0 and step % args.reassign_every == 0:
            reassign_start = time.perf_counter()
            reassign_elbo_fn = None
            if candidate_tree is not None and topm_cache is not None:
                reassign_elbo_fn = sparse_elbo_fn(
                    topm_cache,
                    candidate_tree,
                    args.batch_size,
                    args.candidate_m,
                    args.components,
                    args.top_m,
                )

            prior_model = reassign(
                prior_model,
                model,
                x,
                args.batch_size,
                args.reassign_fraction,
                precision=precision,
                elbo_fn=reassign_elbo_fn,
            )
            reassign_seconds = time.perf_counter() - reassign_start
            rebuild_start = time.perf_counter()
            candidate_tree, topm_cache = build_sparse_index(
                prior_model,
                top_m=args.top_m,
                candidate_m=args.candidate_m,
                precision=precision,
            )
            sparse_rebuild_seconds += time.perf_counter() - rebuild_start
        candidate_indices = None
        kdtree_seconds = 0.0
        if candidate_tree is not None:
            tk = time.perf_counter()
            candidate_indices = query_candidate_indices(
                candidate_tree,
                x[:, :3],
                args.candidate_m,
                args.components,
            )
            kdtree_seconds = time.perf_counter() - tk
        model, prior_stats, space_stats, color_stats, _semantic_stats = fit_gmm_step(
            prior_model,
            model,
            data=x,
            batch_size=args.batch_size,
            prior_stats=prior_stats,
            space_stats=space_stats,
            color_stats=color_stats,
            precision=precision,
            top_m=args.top_m,
            candidate_indices=candidate_indices,
            topm_cache=topm_cache,
        )
        frame_metric = {
            "frame": step,
            "points": int(x.shape[0]),
            "seconds": time.perf_counter() - start,
            "densify_seconds": densify_seconds,
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
            "kdtree_seconds": kdtree_seconds,
            "reassign_seconds": reassign_seconds,
            "sparse_rebuild_seconds": sparse_rebuild_seconds,
            "precision": precision,
        }
        if args.save_every is not None and (step + 1) % args.save_every == 0:
            model_path = output.checkpoint(model, data_params, f"model_{step:03d}.json")
            frame_metric["model"] = str(model_path)
        metrics["frames"].append(frame_metric)

    if args.eval:
        eval_frames = [x for x in eval_iter]
        metrics["eval"] = eval_elbo(
            model,
            eval_frames,
            args.eval_batch_size,
            precision,
            args.eval_subsample,
            args.seed,
            top_m=args.top_m,
            candidate_m=args.candidate_m,
        )

    output.final_model(model, data_params, metrics)
    output.metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
