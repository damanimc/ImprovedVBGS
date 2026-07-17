"""Shared precision compile/load hooks for continual training scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from vbgs.model.jaxpr_precision import DEFAULT_TOLERANCE
from vbgs.model.precision import (
    PrecisionMap,
    compile_op_precision_maps,
    load_op_precision_maps,
    make_sum_stats_probe,
)
from vbgs.model.train import compute_elbo_delta


def precision_batch(normalized, batch_size):
    batch = normalized[:batch_size]
    if batch.shape[0] < batch_size:
        pad = np.zeros((batch_size - batch.shape[0], batch.shape[1]), dtype=batch.dtype)
        batch = np.concatenate([batch, pad], axis=0)
    return jnp.expand_dims(jnp.array(batch), -1)


def _sum_stats_probe_from_model(prior_model, batch):
    _, posteriors = compute_elbo_delta(prior_model, batch)
    cat_i = prior_model.mixture.expand_to_categorical_dims(batch)
    return make_sum_stats_probe(prior_model, posteriors, cat_i)


def ensure_op_precision_runtime(
    *,
    mode: str,
    frame_idx: int,
    search_frame: int,
    prior_model,
    normalized,
    batch_size: int,
    output_dir: Path,
    tolerance: float = DEFAULT_TOLERANCE,
    white_noise_key=None,
    bundle_dir: Path | None = None,
    search_mode: str = "homogeneous",
):
    """Compile or load op-level precision maps for the current frame."""
    if mode not in ("auto", "op"):
        return "fp64" if mode == "auto" else mode, PrecisionMap(
            compute_elbo_delta=mode,
            sum_stats_over_samples=mode,
            tolerance=tolerance,
        )

    bundle_dir = bundle_dir or (output_dir / "precision_maps")
    batch = precision_batch(normalized, batch_size)
    leaf, weights, event_dim = _sum_stats_probe_from_model(prior_model, batch)

    if (bundle_dir / "precision_bundle.json").exists() and (
        mode == "op" or frame_idx >= search_frame
    ):
        bundle = load_op_precision_maps(
            bundle_dir,
            prior_model,
            batch,
            leaf,
            weights,
            event_dim,
        )
        precision_map = PrecisionMap.from_bundle(bundle)
        return "op", precision_map

    if mode == "auto" and frame_idx < search_frame:
        return "fp64", PrecisionMap(tolerance=tolerance)

    if white_noise_key is None:
        white_noise_key = jr.PRNGKey(0)

    bundle = compile_op_precision_maps(
        model=prior_model,
        batch_data=batch,
        sum_stats_leaf=leaf,
        sum_stats_weights=weights,
        sum_stats_event_dim=event_dim,
        tolerance=tolerance,
        output_dir=bundle_dir,
        white_noise_key=white_noise_key,
        search_mode=search_mode,
    )
    precision_map = PrecisionMap.from_bundle(bundle)
    precision_map.save(output_dir / "precision_map.json")
    return "op", precision_map
