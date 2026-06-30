"""Fused weighted sufficient-statistics contraction (paper Algorithm 3)."""

import numpy as np
import jax
import jax.numpy as jnp


def fused_weighted_sum(leaf_array, weights, event_dim: int):
    """Contract sample axes without materialising (B, N, K) intermediates."""
    sample_size = int(leaf_array.shape[0])
    if event_dim == 0:
        leaf = leaf_array.reshape(sample_size, *leaf_array.shape[1:])
        w = weights.reshape(sample_size, *weights.shape[1:])
        return jnp.einsum("s...,s...->...", w, leaf, optimize=True)

    batch_shape = leaf_array.shape[1:-event_dim]
    event_shape = leaf_array.shape[-event_dim:]
    weight_batch_shape = weights.shape[1:-event_dim]
    out_batch_shape = np.broadcast_shapes(batch_shape, weight_batch_shape)
    leaf = leaf_array.reshape(sample_size, *batch_shape, -1)
    w = weights.reshape(
        sample_size,
        *weight_batch_shape,
        *weights.shape[-event_dim:],
    )
    w = jnp.squeeze(w, axis=tuple(range(w.ndim - event_dim, w.ndim)))
    summed = jnp.einsum("s...,s...k->...k", w, leaf, optimize=True)
    return summed.reshape(*out_batch_shape, *event_shape)
