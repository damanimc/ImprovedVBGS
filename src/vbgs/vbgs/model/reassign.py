# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/vbgs/blob/main/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import jax
import jax.numpy as jnp

import equinox as eqx

from vbgs.model.train import compute_elbo_only_delta_with_precision


from vbgs.model.feature_layout import N_COLOR, N_SPATIAL


@jax.jit
def update_initial_model(initial_model, s_means, c_means, sem_means=None):
    initial_model = eqx.tree_at(
        lambda model: model.likelihood._posterior_params.eta.eta_1,
        initial_model,
        replace=s_means * initial_model.likelihood.kappa,
    )
    initial_model = eqx.tree_at(
        lambda model: model.delta._posterior_params.eta.eta_1,
        initial_model,
        replace=c_means * initial_model.delta.kappa,
    )
    if sem_means is not None and initial_model.semantic_delta is not None:
        initial_model = eqx.tree_at(
            lambda model: model.semantic_delta._posterior_params.eta.eta_1,
            initial_model,
            replace=sem_means * initial_model.semantic_delta.kappa,
        )
    return initial_model


def reassign(
    initial_model,
    model,
    data,
    batch_size,
    fraction=0.05,
    debug=False,
    precision="fp64",
    elbo_fn=None,
    point_indices=None,
    point_elbos=None,
):
    """
    Heuristic to force better assignments. Takes n points with the lowest elbo,
    and reassigns them to n components that are currently unused.
    n is determined dynamically as a fraction of the unused components.
    """

    alpha = np.asarray(jax.block_until_ready(model.prior.alpha))
    unused = alpha <= initial_model.prior.prior_alpha.min().item()
    available = int(unused.sum())
    max_reassign = int(alpha.shape[0] * fraction)
    requested_reassign = int(available * fraction)
    if available <= 0 or max_reassign <= 0 or requested_reassign <= 0:
        return initial_model

    if point_indices is not None:
        candidates = np.asarray(point_indices, dtype=np.int64)
        if candidates.size == 0:
            return initial_model
        n_reassign = min(max_reassign, requested_reassign, int(candidates.size))
        if candidates.size > n_reassign:
            if point_elbos is not None:
                candidate_elbos = np.asarray(point_elbos)
                p_elbo = -candidate_elbos
                p_elbo = p_elbo - p_elbo.min()
                total = p_elbo.sum()
                p_elbo = None if total <= 0 else p_elbo / total
            else:
                p_elbo = None
            point_idcs = np.random.choice(
                candidates,
                p=p_elbo,
                size=n_reassign,
                replace=False,
            )
        else:
            point_idcs = candidates
        elbos = point_elbos
    else:
        elbo_batches = []
        n_batches = int(np.ceil(data.shape[0] / batch_size))
        for batch_idx in range(n_batches):
            raw_xi = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            if elbo_fn is not None:
                elbo_batches.append(elbo_fn(raw_xi))
                continue

            xi = raw_xi
            xi = jnp.expand_dims(jnp.array(xi), -1)

            size = xi.shape[0]
            if size < batch_size:
                # Pad to keep one compiled shape.
                xi = jnp.concatenate(
                    [xi, jnp.zeros((batch_size - size, *xi.shape[1:]))],
                    axis=0,
                )
                elbo = compute_elbo_only_delta_with_precision(
                    initial_model, xi, precision
                )
                elbo = elbo[:size]
            else:
                elbo = compute_elbo_only_delta_with_precision(
                    initial_model, xi, precision
                )
            elbo_batches.append(elbo)

        elbos = np.concatenate(
            [np.asarray(jax.block_until_ready(elbo)) for elbo in elbo_batches]
        )
        p_elbo = -elbos
        p_elbo = p_elbo - p_elbo.min()  # smallest value 0
        total = p_elbo.sum()
        p_elbo = None if total <= 0 else p_elbo / total
        point_idcs = np.random.choice(
            np.arange(len(elbos)),
            p=p_elbo,
            size=min(max_reassign, requested_reassign, int(len(elbos))),
            replace=False,
        )
        n_reassign = int(point_idcs.shape[0])

    if n_reassign <= 0:
        return initial_model

    # Keep JAX scatter shapes fixed across frames; invalid padded slots write no-op values.
    pad_component = int(np.argmax(alpha))
    component_idcs = np.full(max_reassign, pad_component, dtype=np.int64)
    component_idcs[:n_reassign] = np.flatnonzero(unused)[
        np.argsort(alpha[unused])[:n_reassign]
    ]
    padded_point_idcs = np.zeros(max_reassign, dtype=np.int64)
    padded_point_idcs[:n_reassign] = np.asarray(point_idcs, dtype=np.int64)[:n_reassign]
    valid_mask = jnp.asarray(np.arange(max_reassign) < n_reassign)

    # Move unused component means to low-ELBO points before the next update.
    s_means = initial_model.likelihood.mean
    old_s_targets = s_means[component_idcs]
    new_s_targets = jnp.where(
        valid_mask[:, None, None],
        data[padded_point_idcs, :N_SPATIAL, jnp.newaxis],
        old_s_targets,
    )
    s_means = s_means.at[component_idcs].set(
        new_s_targets
    )

    c_means = initial_model.delta.mean
    old_c_targets = c_means[component_idcs]
    new_c_targets = jnp.where(
        valid_mask[:, None, None],
        data[padded_point_idcs, N_SPATIAL : N_SPATIAL + N_COLOR, jnp.newaxis],
        old_c_targets,
    )
    c_means = c_means.at[component_idcs].set(
        new_c_targets
    )

    sem_means = None
    if initial_model.semantic_delta is not None:
        sem_means = initial_model.semantic_delta.mean
        old_sem_targets = sem_means[component_idcs]
        new_sem_targets = jnp.where(
            valid_mask[:, None, None],
            data[padded_point_idcs, N_SPATIAL + N_COLOR :, jnp.newaxis],
            old_sem_targets,
        )
        sem_means = sem_means.at[component_idcs].set(
            new_sem_targets
        )

    initial_model = update_initial_model(
        initial_model, s_means, c_means, sem_means
    )

    if debug:
        # plot_selection(elbos, point_idcs, data[:, 3:].reshape((512, 512, 3)))
        return initial_model, {"elbo": elbos, "point_indices": point_idcs[:n_reassign]}
    else:
        return initial_model
