"""Shared sparse ELBO evaluation helpers."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from vbgs.model.continual import build_sparse_index, query_candidate_indices
from vbgs.model.train import (
    compute_candidate_topm_elbo_cached,
    compute_elbo_only_delta_with_precision,
)


def _pad_batch(xi, batch_size, candidate_idx=None):
    size = xi.shape[0]
    xi = np.expand_dims(np.asarray(xi), -1)
    if size >= batch_size:
        return xi, candidate_idx, size
    xi = np.concatenate(
        [xi, np.zeros((batch_size - size, *xi.shape[1:]), dtype=xi.dtype)],
        axis=0,
    )
    if candidate_idx is not None:
        candidate_idx = np.concatenate(
            [
                candidate_idx,
                np.zeros(
                    (batch_size - size, candidate_idx.shape[1]),
                    dtype=candidate_idx.dtype,
                ),
            ],
            axis=0,
        )
    return xi, candidate_idx, size


def sparse_elbo_fn(topm_cache, candidate_tree, batch_size, candidate_m, n_components, top_m):
    """Build an ELBO callable for sparse reassign scoring."""

    def elbo_fn(raw_xi):
        candidate_idx = query_candidate_indices(
            candidate_tree,
            raw_xi[:, :3],
            candidate_m,
            n_components,
        )
        xi, candidate_idx, size = _pad_batch(raw_xi, batch_size, candidate_idx)
        elbo = compute_candidate_topm_elbo_cached(
            topm_cache.space,
            topm_cache.color,
            topm_cache.semantic,
            topm_cache.prior_log_mean,
            topm_cache.mean,
            topm_cache.inv_sigma,
            jnp.asarray(xi),
            jnp.asarray(candidate_idx),
            int(top_m),
            int(topm_cache.space_dim),
            int(topm_cache.color_dim),
            int(topm_cache.semantic_dim),
            int(topm_cache.n_sem),
        )
        return jnp.asarray(np.asarray(jax.block_until_ready(elbo))[:size])

    return elbo_fn


def eval_elbo(
    model,
    frames,
    batch_size,
    precision,
    subsample,
    seed,
    top_m=None,
    candidate_m=None,
):
    """Evaluate mean ELBO per frame, optionally with sparse top-M inference."""
    rng = np.random.default_rng(seed)
    sparse_eval = top_m is not None and candidate_m is not None
    eval_tree = None
    eval_cache = None
    if sparse_eval:
        eval_tree, eval_cache = build_sparse_index(
            model,
            top_m=top_m,
            candidate_m=candidate_m,
            precision=precision,
        )
    results = []
    for eval_idx, x in enumerate(frames):
        if subsample is not None and subsample > 0 and x.shape[0] > subsample:
            x = x[rng.choice(x.shape[0], size=subsample, replace=False)]
        start = time.perf_counter()
        kdtree_seconds = 0.0
        values = []
        for batch_idx in range(int(np.ceil(x.shape[0] / batch_size))):
            xi = x[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            candidate_idx = None
            if sparse_eval:
                tk = time.perf_counter()
                candidate_idx = query_candidate_indices(
                    eval_tree,
                    xi[:, :3],
                    candidate_m,
                    model.mixture.prior.alpha.shape[0],
                )
                kdtree_seconds += time.perf_counter() - tk
            xi, candidate_idx, size = _pad_batch(xi, batch_size, candidate_idx)
            if sparse_eval:
                elbo = compute_candidate_topm_elbo_cached(
                    eval_cache.space,
                    eval_cache.color,
                    eval_cache.semantic,
                    eval_cache.prior_log_mean,
                    eval_cache.mean,
                    eval_cache.inv_sigma,
                    jnp.asarray(xi),
                    jnp.asarray(candidate_idx),
                    int(top_m),
                    int(eval_cache.space_dim),
                    int(eval_cache.color_dim),
                    int(eval_cache.semantic_dim),
                    int(eval_cache.n_sem),
                )
            else:
                elbo = compute_elbo_only_delta_with_precision(model, xi, precision)
            values.append(np.asarray(jax.block_until_ready(elbo))[:size])
        values = np.concatenate(values)
        results.append(
            {
                "eval_frame": eval_idx,
                "points": int(x.shape[0]),
                "seconds": time.perf_counter() - start,
                "kdtree_seconds": kdtree_seconds,
                "mean_elbo": float(values.mean()),
            }
        )
    return results
