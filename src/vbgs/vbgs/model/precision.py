"""Precision map orchestration for VBGS training (function- and op-level)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.tree_util as jtu
import jax.numpy as jnp

from vbgs.model.jaxpr_precision import DEFAULT_TOLERANCE
from vbgs.model.precision_runtime import (
    PrecisionBundle,
    compile_precision_bundle,
    load_precision_bundle,
    set_precision_runtime,
)


@dataclass(frozen=True)
class PrecisionMap:
    """Flat, JSON-serializable summary of the selected precision for metrics."""

    compute_elbo_delta: str = "fp64"
    sum_stats_over_samples: str = "fp64"
    tolerance: float = DEFAULT_TOLERANCE
    posterior_relative_error: float | None = None
    elbo_relative_error: float | None = None
    fp64_seconds: float | None = None
    fp32_seconds: float | None = None
    tf32_seconds: float | None = None
    selected_reason: str = "baseline"
    precision_aware_pass: dict | None = None
    structure_aware_pass: dict | None = None
    latency_aware_pass: dict | None = None
    mode: str = "function"

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_bundle(cls, bundle: PrecisionBundle):
        elbo_err = None
        if bundle.elbo_map is not None:
            elbo_err = bundle.elbo_map.elbo_relative_error
        latency = (
            bundle.elbo_map.latency_aware_pass if bundle.elbo_map is not None else None
        )
        return cls(
            compute_elbo_delta="op",
            sum_stats_over_samples="op",
            tolerance=bundle.tolerance,
            posterior_relative_error=elbo_err,
            elbo_relative_error=elbo_err,
            fp64_seconds=(latency or {}).get("fp64_seconds"),
            fp32_seconds=(latency or {}).get("optimized_seconds"),
            selected_reason=bundle.elbo_map.selected_reason
            if bundle.elbo_map is not None
            else "op-level map",
            precision_aware_pass=bundle.elbo_map.precision_aware_pass
            if bundle.elbo_map is not None
            else None,
            structure_aware_pass=bundle.elbo_map.structure_aware_pass
            if bundle.elbo_map is not None
            else None,
            latency_aware_pass=latency,
            mode="op",
        )


def dominant_precision(op_map) -> str:
    if op_map is None or not op_map.equation_precisions:
        return "fp64"
    counts: dict[str, int] = {}
    for value in op_map.equation_precisions.values():
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def compile_op_precision_maps(
    model,
    batch_data,
    sum_stats_leaf,
    sum_stats_weights,
    sum_stats_event_dim: int,
    tolerance: float = DEFAULT_TOLERANCE,
    output_dir: str | Path | None = None,
    white_noise_key=None,
) -> PrecisionBundle:
    return compile_precision_bundle(
        model=model,
        batch_data=batch_data,
        sum_stats_leaf=sum_stats_leaf,
        sum_stats_weights=sum_stats_weights,
        sum_stats_event_dim=sum_stats_event_dim,
        tolerance=tolerance,
        output_dir=output_dir,
        white_noise_key=white_noise_key,
        search=True,
    )


def load_op_precision_maps(
    directory: str | Path,
    model,
    batch_data,
    sum_stats_leaf,
    sum_stats_weights,
    sum_stats_event_dim: int,
) -> PrecisionBundle:
    return load_precision_bundle(
        directory,
        model,
        batch_data,
        sum_stats_leaf,
        sum_stats_weights,
        sum_stats_event_dim,
    )


def clear_precision_runtime():
    set_precision_runtime(None)


def make_sum_stats_probe(initial_model, posteriors, cat_i):
    """Build representative tensors for sum-stats jaxpr tracing."""
    likelihood = initial_model.mixture.likelihood
    x = cat_i[:, :, :-3]
    counts_shape = likelihood.get_sample_shape(x) + likelihood.get_batch_shape(x)
    shape = counts_shape + (1,) * likelihood.event_dim
    counts = jnp.ones(counts_shape)
    weights = (
        likelihood.expand_event_dims(posteriors)
        if posteriors is not None
        else jnp.ones(shape)
    )
    likelihood_stats = likelihood.likelihood.statistics(x)
    param_stats = likelihood.map_stats_to_params(likelihood_stats, counts)
    leaf = next(iter(jtu.tree_leaves(param_stats)))
    return leaf, weights, likelihood.event_dim
