"""Densify under-constructed regions during continual VBGS training."""

import numpy as np
import jax.numpy as jnp

from vbgs.model.feature_layout import N_COLOR, N_SPATIAL
from vbgs.model.reassign import update_initial_model

DENSIFY_FRACTION = 0.05
SOFT_MANHATTAN_SCALE = 1.0
DENSIFY_DISTANCE_CHUNK = 8192


def soft_manhattan_distance(points_xyz, component_xyz, scale=1.0):
    """Return soft-min L1 distance from each point to candidate components."""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    component_xyz = np.asarray(component_xyz, dtype=np.float32)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")

    distances = np.abs(points_xyz[:, None, :] - component_xyz).sum(axis=-1)
    n_candidates = component_xyz.shape[-2]
    values = -scale * distances
    max_values = values.max(axis=-1, keepdims=True)
    lse = np.squeeze(max_values, axis=-1) + np.log(
        np.exp(values - max_values).sum(axis=-1)
    )
    return -((lse - np.log(n_candidates)) / scale)


def chunked_soft_manhattan_distance(
    points_xyz,
    means_xyz,
    candidate_indices=None,
    scale=1.0,
    chunk_size=DENSIFY_DISTANCE_CHUNK,
):
    """Compute densify scores without materializing all point-candidate pairs."""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    means_xyz = np.asarray(means_xyz, dtype=np.float32)
    scores = np.empty((points_xyz.shape[0],), dtype=np.float32)
    for start in range(0, points_xyz.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), points_xyz.shape[0])
        points_chunk = points_xyz[start:stop]
        if candidate_indices is None:
            component_chunk = means_xyz[None, :, :]
        else:
            component_chunk = means_xyz[np.asarray(candidate_indices[start:stop])]
        scores[start:stop] = soft_manhattan_distance(
            points_chunk,
            component_chunk,
            scale=scale,
        )
    return scores


def densify_candidate_indices(candidate_tree, points_xyz, candidate_m):
    """Query spatial candidate components for densification scoring."""
    if candidate_tree is None or candidate_m is None:
        return None

    k = min(int(candidate_m), int(candidate_tree.n))
    candidate_indices = candidate_tree.query(points_xyz, k=k, workers=-1)[1]
    if candidate_indices.ndim == 1:
        candidate_indices = candidate_indices[:, None]
    return candidate_indices


def _densify_stats(
    n_available,
    n_strict_unused,
    component_offset,
    recycle_mode,
    reassign_fraction,
    *,
    densify_points=0,
    densify_distance_threshold=None,
    densify_distance_mean=None,
    unseen_distance_threshold=None,
    unseen_points=0,
    unseen_fraction=0.0,
    densify_skipped_reason=None,
):
    return {
        "available_components": n_available,
        "strict_unused_components": n_strict_unused,
        "component_offset": component_offset,
        "densify_points": densify_points,
        "densify_distance_threshold": densify_distance_threshold,
        "densify_distance_mean": densify_distance_mean,
        "unseen_distance_threshold": unseen_distance_threshold,
        "unseen_points": unseen_points,
        "unseen_fraction": unseen_fraction,
        "densify_skipped_reason": densify_skipped_reason,
        "recycle_mode": recycle_mode,
        "reassign_fraction": reassign_fraction if recycle_mode else None,
    }


def select_densify_points(
    initial_model,
    model,
    data,
    candidate_indices=None,
    component_offset=0,
    vacant_mask=None,
    point_ratio=None,
    unseen_distance_threshold=None,
    min_unseen_fraction=0.0,
    min_unseen_points=0,
    recycle_if_full=False,
    reassign_fraction=0.05,
):
    """Select current-frame points for insertion into vacant component slots."""
    strict_unused = np.asarray(
        model.prior.alpha <= initial_model.prior.prior_alpha.min().item()
    )
    if vacant_mask is not None:
        strict_unused = strict_unused & np.asarray(vacant_mask, dtype=bool)
    n_strict_unused = int(strict_unused.sum())
    recycle_mode = bool(recycle_if_full and n_strict_unused == 0)
    n_available = int(model.prior.alpha.shape[0]) if recycle_mode else n_strict_unused
    component_offset = max(0, int(component_offset))
    n_remaining = max(0, n_available - component_offset)
    if recycle_mode:
        fraction = max(0.0, float(reassign_fraction))
        requested = int(np.ceil(n_available * fraction)) if fraction > 0 else 0
    elif point_ratio is not None:
        ratio = max(0.0, float(point_ratio))
        requested = int(np.ceil(data.shape[0] * ratio)) if ratio > 0 else 0
    else:
        requested = max(1, int(n_available * DENSIFY_FRACTION)) if n_remaining > 0 else 0
    n_select = min(requested, n_remaining)
    if n_select <= 0 or data.shape[0] == 0:
        return np.asarray([], dtype=np.int64), _densify_stats(
            n_available,
            n_strict_unused,
            component_offset,
            recycle_mode,
            reassign_fraction,
            unseen_distance_threshold=unseen_distance_threshold,
            densify_skipped_reason="no_available_components",
        )

    points_xyz = np.asarray(data[:, :N_SPATIAL])
    means_xyz = np.asarray(initial_model.likelihood.mean[:, :N_SPATIAL, 0])

    distances = chunked_soft_manhattan_distance(
        points_xyz,
        means_xyz,
        candidate_indices=candidate_indices,
        scale=SOFT_MANHATTAN_SCALE,
    )
    if unseen_distance_threshold is not None:
        unseen_mask = distances >= float(unseen_distance_threshold)
    else:
        unseen_mask = np.ones((distances.shape[0],), dtype=bool)
    unseen_idcs = np.flatnonzero(unseen_mask)
    unseen_points = int(unseen_idcs.shape[0])
    unseen_fraction = float(unseen_points / distances.shape[0])
    if (
        unseen_points < int(min_unseen_points)
        or unseen_fraction < float(min_unseen_fraction)
    ):
        return np.asarray([], dtype=np.int64), _densify_stats(
            n_available,
            n_strict_unused,
            component_offset,
            recycle_mode,
            reassign_fraction,
            densify_distance_mean=float(distances.mean()),
            unseen_distance_threshold=unseen_distance_threshold,
            unseen_points=unseen_points,
            unseen_fraction=unseen_fraction,
            densify_skipped_reason="insufficient_unseen_area",
        )

    n_select = min(n_select, unseen_points)
    point_idcs = unseen_idcs[np.argsort(distances[unseen_idcs])[-n_select:]]

    return point_idcs.astype(np.int64), _densify_stats(
        n_available,
        n_strict_unused,
        component_offset,
        recycle_mode,
        reassign_fraction,
        densify_points=int(point_idcs.shape[0]),
        densify_distance_threshold=(
            float(distances[point_idcs].min()) if n_select else None
        ),
        densify_distance_mean=float(distances.mean()),
        unseen_distance_threshold=unseen_distance_threshold,
        unseen_points=unseen_points,
        unseen_fraction=unseen_fraction,
    )


def densify_from_frame(
    initial_model,
    model,
    data,
    candidate_indices=None,
    component_offset=0,
    vacant_mask=None,
    point_ratio=None,
    unseen_distance_threshold=None,
    min_unseen_fraction=0.0,
    min_unseen_points=0,
    recycle_if_full=False,
    reassign_fraction=0.05,
    debug=False,
    return_indices=False,
):
    """Initialize vacant Gaussian slots in under-constructed current-frame regions."""
    point_idcs, stats = select_densify_points(
        initial_model,
        model,
        data,
        candidate_indices=candidate_indices,
        component_offset=component_offset,
        vacant_mask=vacant_mask,
        point_ratio=point_ratio,
        unseen_distance_threshold=unseen_distance_threshold,
        min_unseen_fraction=min_unseen_fraction,
        min_unseen_points=min_unseen_points,
        recycle_if_full=recycle_if_full,
        reassign_fraction=reassign_fraction,
    )
    n_insert = int(point_idcs.shape[0])
    stats["densified_components"] = n_insert
    if n_insert <= 0:
        if return_indices:
            return (initial_model, stats, np.asarray([], dtype=np.int64))
        return (initial_model, stats) if debug else initial_model

    strict_unused = np.asarray(
        model.prior.alpha <= initial_model.prior.prior_alpha.min().item()
    )
    if vacant_mask is not None:
        strict_unused = strict_unused & np.asarray(vacant_mask, dtype=bool)
    vacant_components = np.flatnonzero(strict_unused)
    if vacant_components.shape[0] == 0 and stats.get("recycle_mode", False):
        vacant_components = np.arange(model.prior.alpha.shape[0])
    vacant_components = vacant_components[
        np.argsort(np.asarray(model.prior.alpha)[vacant_components])
    ]
    component_offset = max(0, int(component_offset))
    component_idcs = vacant_components[component_offset : component_offset + n_insert]
    n_insert = min(n_insert, int(component_idcs.shape[0]))
    point_idcs = point_idcs[:n_insert]
    stats["densified_components"] = n_insert
    if n_insert <= 0:
        if return_indices:
            return (initial_model, stats, np.asarray([], dtype=np.int64))
        return (initial_model, stats) if debug else initial_model

    s_means = initial_model.likelihood.mean
    s_means = s_means.at[component_idcs].set(
        data[point_idcs, :N_SPATIAL, jnp.newaxis]
    )

    c_means = initial_model.delta.mean
    c_means = c_means.at[component_idcs].set(
        data[point_idcs, N_SPATIAL : N_SPATIAL + N_COLOR, jnp.newaxis]
    )

    sem_means = None
    if initial_model.semantic_delta is not None:
        sem_means = initial_model.semantic_delta.mean
        sem_means = sem_means.at[component_idcs].set(
            data[point_idcs, N_SPATIAL + N_COLOR :, jnp.newaxis]
        )

    initial_model = update_initial_model(initial_model, s_means, c_means, sem_means)
    if return_indices:
        return initial_model, stats, component_idcs.astype(np.int64)
    return (initial_model, stats) if debug else initial_model
