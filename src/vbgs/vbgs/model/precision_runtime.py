"""Runtime wiring for operation-level precision maps."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

from vbgs.model.jaxpr_precision import (
    DEFAULT_TOLERANCE,
    OpPrecisionMap,
    TracedFunction,
    build_jitted_runner,
    search_op_precision_map,
    trace_function,
    white_noise_batch,
)
from vbgs.model.train import _compute_elbo_delta_impl, _compute_elbo_only_delta_impl
from vbgs.vi.sum_stats_fused import fused_weighted_sum


@dataclass
class PrecisionBundle:
    tolerance: float = DEFAULT_TOLERANCE
    elbo_map: OpPrecisionMap | None = None
    elbo_only_map: OpPrecisionMap | None = None
    sum_stats_map: OpPrecisionMap | None = None
    elbo_traced: TracedFunction | None = field(default=None, repr=False)
    elbo_only_traced: TracedFunction | None = field(default=None, repr=False)
    sum_stats_traced: TracedFunction | None = field(default=None, repr=False)
    elbo_runner: object | None = field(default=None, repr=False)
    elbo_only_runner: object | None = field(default=None, repr=False)
    sum_stats_runner: object | None = field(default=None, repr=False)

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        with open(directory / "precision_bundle.json") as f:
            meta = json.load(f)
        bundle = cls(tolerance=meta.get("tolerance", DEFAULT_TOLERANCE))
        elbo_path = directory / "compute_elbo_delta.opmap.json"
        elbo_only_path = directory / "compute_elbo_only_delta.opmap.json"
        sum_path = directory / "sum_stats_over_samples.opmap.json"
        if elbo_path.exists():
            bundle.elbo_map = OpPrecisionMap.load(elbo_path)
        if elbo_only_path.exists():
            bundle.elbo_only_map = OpPrecisionMap.load(elbo_only_path)
        if sum_path.exists():
            bundle.sum_stats_map = OpPrecisionMap.load(sum_path)
        return bundle

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"tolerance": self.tolerance}
        with open(directory / "precision_bundle.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self.elbo_map is not None:
            self.elbo_map.save(directory / "compute_elbo_delta.opmap.json")
        if self.elbo_only_map is not None:
            self.elbo_only_map.save(directory / "compute_elbo_only_delta.opmap.json")
        if self.sum_stats_map is not None:
            self.sum_stats_map.save(directory / "sum_stats_over_samples.opmap.json")

    def is_ready(self) -> bool:
        return self.elbo_runner is not None and self.sum_stats_runner is not None


_RUNTIME: PrecisionBundle | None = None


def get_precision_runtime() -> PrecisionBundle | None:
    return _RUNTIME


def set_precision_runtime(bundle: PrecisionBundle | None):
    global _RUNTIME
    _RUNTIME = bundle


def _trace_elbo(model, data) -> TracedFunction:
    return trace_function("compute_elbo_delta", _compute_elbo_delta_impl, model, data)


def _trace_elbo_only(model, data) -> TracedFunction:
    return trace_function(
        "compute_elbo_only_delta", _compute_elbo_only_delta_impl, model, data
    )


def _trace_sum_stats(example_leaf, example_weights, event_dim: int) -> TracedFunction:
    return trace_function(
        "sum_stats_over_samples",
        lambda leaf, weights: fused_weighted_sum(leaf, weights, event_dim),
        example_leaf,
        example_weights,
    )


def _build_runners(bundle: PrecisionBundle):
    if bundle.elbo_traced is not None and bundle.elbo_map is not None:
        bundle.elbo_runner = build_jitted_runner(bundle.elbo_traced, bundle.elbo_map)
    if bundle.elbo_only_traced is not None:
        only_map = bundle.elbo_only_map or bundle.elbo_map
        if only_map is not None:
            bundle.elbo_only_runner = build_jitted_runner(
                bundle.elbo_only_traced, only_map
            )
    if bundle.sum_stats_traced is not None and bundle.sum_stats_map is not None:
        bundle.sum_stats_runner = build_jitted_runner(
            bundle.sum_stats_traced, bundle.sum_stats_map
        )


def compile_precision_bundle(
    model,
    batch_data,
    sum_stats_leaf,
    sum_stats_weights,
    sum_stats_event_dim: int,
    tolerance: float = DEFAULT_TOLERANCE,
    output_dir: str | Path | None = None,
    white_noise_key=None,
    search: bool = True,
) -> PrecisionBundle:
    if white_noise_key is None:
        white_noise_key = jr.PRNGKey(0)

    noise = white_noise_batch(batch_data.shape, white_noise_key, dtype=batch_data.dtype)
    bundle = PrecisionBundle(tolerance=tolerance)
    bundle.elbo_traced = _trace_elbo(model, noise)
    bundle.elbo_only_traced = _trace_elbo_only(model, noise)

    leaf_shape = sum_stats_leaf.shape
    weight_shape = sum_stats_weights.shape
    noise_leaf = white_noise_batch(
        leaf_shape, jr.fold_in(white_noise_key, 1), dtype=sum_stats_leaf.dtype
    )
    noise_weights = jnp.abs(
        white_noise_batch(
            weight_shape, jr.fold_in(white_noise_key, 2), dtype=sum_stats_weights.dtype
        )
    )
    bundle.sum_stats_traced = _trace_sum_stats(
        noise_leaf, noise_weights, sum_stats_event_dim
    )

    if search:
        elbo_path = None
        elbo_only_path = None
        sum_path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            elbo_path = output_dir / "compute_elbo_delta.opmap.json"
            elbo_only_path = output_dir / "compute_elbo_only_delta.opmap.json"
            sum_path = output_dir / "sum_stats_over_samples.opmap.json"
        bundle.elbo_map = search_op_precision_map(
            bundle.elbo_traced, (model, noise), tolerance=tolerance, path=elbo_path
        )
        bundle.elbo_only_map = search_op_precision_map(
            bundle.elbo_only_traced,
            (model, noise),
            tolerance=tolerance,
            path=elbo_only_path,
        )
        bundle.sum_stats_map = search_op_precision_map(
            bundle.sum_stats_traced,
            (noise_leaf, noise_weights),
            tolerance=tolerance,
            path=sum_path,
        )
    if output_dir is not None:
        bundle.save(output_dir)
    _build_runners(bundle)
    set_precision_runtime(bundle)
    return bundle


def load_precision_bundle(
    directory: str | Path,
    model,
    batch_data,
    sum_stats_leaf,
    sum_stats_weights,
    sum_stats_event_dim: int,
) -> PrecisionBundle:
    bundle = PrecisionBundle.load(directory)
    bundle.elbo_traced = _trace_elbo(model, batch_data)
    bundle.elbo_only_traced = _trace_elbo_only(model, batch_data)
    bundle.sum_stats_traced = _trace_sum_stats(
        sum_stats_leaf, sum_stats_weights, sum_stats_event_dim
    )
    _build_runners(bundle)
    set_precision_runtime(bundle)
    return bundle


def run_elbo_delta(model, data):
    bundle = get_precision_runtime()
    if bundle is None or bundle.elbo_runner is None:
        from vbgs.model.train import compute_elbo_delta

        return compute_elbo_delta(model, data)
    outs = bundle.elbo_runner(model, data)
    if isinstance(outs, (list, tuple)):
        if len(outs) == 2:
            return outs[0], outs[1]
        return outs[0]
    return outs


def run_elbo_only_delta(model, data):
    bundle = get_precision_runtime()
    if bundle is None or bundle.elbo_only_runner is None:
        from vbgs.model.train import compute_elbo_only_delta

        return compute_elbo_only_delta(model, data)
    outs = bundle.elbo_only_runner(model, data)
    if isinstance(outs, (list, tuple)):
        return outs[0]
    return outs


def run_fused_sum_stats(leaf_array, weights, event_dim: int):
    bundle = get_precision_runtime()
    if bundle is None or bundle.sum_stats_runner is None:
        return fused_weighted_sum(leaf_array, weights, event_dim)
    return bundle.sum_stats_runner(leaf_array, weights)
