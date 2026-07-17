"""Shared helpers for continual VBGS training scripts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from vbgs.model.train import build_topm_cache
from vbgs.model.utils import random_mean_init


def build_sparse_index(
    index_model,
    top_m=None,
    candidate_m=None,
    precision="fp64",
    use_topm_cache=True,
):
    """Build spatial and top-M caches used by sparse E-steps."""
    if top_m is None or candidate_m is None:
        return None, None
    means = np.ascontiguousarray(
        np.asarray(index_model.mixture.likelihood.mean[:, :3, 0], dtype=np.float64)
    )
    if not use_topm_cache:
        return cKDTree(means), None
    with ThreadPoolExecutor(max_workers=2) as pool:
        tree_future = pool.submit(cKDTree, means)
        cache_future = pool.submit(build_topm_cache, index_model, precision=precision)
        return tree_future.result(), cache_future.result()


def query_candidate_indices(
    candidate_tree,
    points_xyz,
    candidate_m,
    n_components,
    eps=0.0,
):
    """Query candidate Gaussian indices for each point."""
    if candidate_tree is None or candidate_m is None:
        return None
    k = min(int(candidate_m), int(n_components))
    candidate_indices = candidate_tree.query(
        np.asarray(points_xyz),
        k=k,
        eps=float(eps),
        workers=-1,
    )[1]
    if candidate_indices.ndim == 1:
        candidate_indices = candidate_indices[:, None]
    return candidate_indices


def reset_component_stats(stats, component_idcs):
    """Zero accumulated sufficient statistics for relocated components."""
    if stats is None or len(component_idcs) == 0:
        return stats
    component_idcs = jnp.asarray(component_idcs)

    def reset_leaf(leaf):
        if not hasattr(leaf, "shape") or len(leaf.shape) == 0:
            return leaf
        if leaf.shape[0] < int(component_idcs.max()) + 1:
            return leaf
        return leaf.at[component_idcs].set(jnp.zeros_like(leaf[component_idcs]))

    return jax.tree_util.tree_map(reset_leaf, stats)


def init_source_metrics(init_data, n_components, init_first_frame, init_noise):
    """Describe initialization source semantics without overloaded labels."""
    init_points = int(init_data.shape[0]) if init_data is not None else None
    exact_source_points = (
        init_points
        if init_first_frame and init_points is not None and n_components >= init_points
        else 0
    )
    sampled_with_replacement = bool(
        init_data is not None
        and not (
            init_first_frame
            and init_points is not None
            and n_components == init_points
        )
    )
    return {
        "init_source_points": init_points,
        "init_exact_source_points": exact_source_points,
        "init_sampled_with_replacement": sampled_with_replacement,
        "init_noise": bool(init_noise),
    }


def initialize_mean(
    key,
    init_data,
    n_components,
    event_shape,
    init_random,
    init_first_frame=False,
    add_noise=True,
):
    """Initialize component means and enforce exact first-frame slots when possible."""
    mean_init = random_mean_init(
        key=key,
        x=init_data,
        component_shape=(int(n_components),),
        event_shape=event_shape,
        init_random=init_random,
        add_noise=add_noise,
    )
    init_points = int(init_data.shape[0]) if init_data is not None else None
    if init_first_frame and init_data is not None and n_components >= init_points:
        mean_init = mean_init.at[:init_points].set(
            jnp.asarray(init_data).reshape((-1, *event_shape))
        )
    metrics = init_source_metrics(
        init_data,
        n_components,
        init_first_frame=init_first_frame,
        init_noise=add_noise,
    )
    return mean_init, metrics


def default_densify_stats(unseen_distance_threshold=None):
    """Return complete densify metrics payload for frames with no densify work."""
    return {
        "available_components": 0,
        "strict_unused_components": 0,
        "component_offset": 0,
        "densify_points": 0,
        "densify_distance_threshold": None,
        "densify_distance_mean": None,
        "unseen_distance_threshold": unseen_distance_threshold,
        "unseen_points": 0,
        "unseen_fraction": 0.0,
        "densify_skipped_reason": None,
        "recycle_mode": False,
        "reassign_fraction": None,
        "densified_components": 0,
    }


def strict_unused_count(model, initial_model):
    """Count components still at initial prior mass."""
    return int(
        np.asarray(
            model.prior.alpha <= initial_model.prior.prior_alpha.min().item()
        ).sum()
    )
