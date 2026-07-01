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


import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.nn import softmax
from jax.scipy.special import logsumexp

import numpy as np

from vbgs.vi import utils
from vbgs.vi.utils import ArrayDict
from vbgs.model.feature_layout import model_n_semantic, split_features


def _elbo_logprob_parts(model, d):
    n_sem = model_n_semantic(model)
    ds, dc, dsem = split_features(d, n_sem)
    space_logprob = model.mixture.likelihood.expected_log_likelihood(ds)
    color_logprob = model.delta.expected_log_likelihood(dc)
    logprob = space_logprob + color_logprob
    if dsem is not None:
        logprob = logprob + model.semantic_delta.expected_log_likelihood(dsem)
    prior_logprob = model.mixture.prior.log_mean()
    logprob = logprob + prior_logprob
    return logprob


def _elbo_kl_parts(model, mixdims):
    prior_kl = model.mixture.prior.kl_divergence()
    space_kl = model.mixture.likelihood.kl_divergence().sum(mixdims)
    color_kl = model.delta.kl_divergence().sum(mixdims)
    kl = space_kl + prior_kl + color_kl
    if model.semantic_delta is not None:
        kl = kl + model.semantic_delta.kl_divergence().sum(mixdims)
    return kl


def _cast_floating_tree(tree, dtype):
    return jtu.tree_map(
        lambda x: x.astype(dtype)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        tree,
    )


def get_likelihood_sst(self, data, weights):
    """
    Computes the sufficient statistics for the likelihood
    """
    x = data[0] if isinstance(data, tuple) else data

    counts_shape = self.get_sample_shape(x) + self.get_batch_shape(x)
    shape = counts_shape + (1,) * self.event_dim
    counts = jnp.ones(counts_shape)
    sample_dims = self.get_sample_dims(x)

    weights = (
        self.expand_event_dims(weights)
        if weights is not None
        else jnp.ones(shape)
    )

    likelihood_stats = self.likelihood.statistics(data)

    param_stats = self.map_stats_to_params(likelihood_stats, counts)
    summed_stats = self.sum_stats_over_samples(
        param_stats, weights, sample_dims
    )

    return summed_stats, {
        "likelihood_stats": likelihood_stats,
        "counts": counts,
        "param_stats": param_stats,
        "weights": weights,
        "summmed_stats": summed_stats,
    }


def fit_gmm(initial_model, model, data):
    data = jnp.expand_dims(data, -1)
    d = model.mixture.expand_to_categorical_dims(data)
    n_sem = model_n_semantic(model)
    ds, dc, dsem = split_features(d, n_sem)
    space_logprob = model.mixture.likelihood.expected_log_likelihood(ds)
    color_logprob = model.delta.expected_log_likelihood(dc)
    prior_logprob = model.mixture.prior.log_mean()
    logprob = space_logprob + color_logprob + prior_logprob
    if dsem is not None:
        logprob = logprob + model.semantic_delta.expected_log_likelihood(dsem)
    mixdims = tuple(range(-model.mixture.prior.event_dim, 0))
    posteriors = softmax(logprob, mixdims)

    cat_i = initial_model.mixture.expand_to_categorical_dims(data)
    ps = initial_model.mixture._to_stats(
        posteriors, initial_model.mixture.get_sample_dims(cat_i)
    )
    ss, _ = get_likelihood_sst(
        initial_model.mixture.likelihood, ds, posteriors
    )
    cs, _ = get_likelihood_sst(initial_model.delta, dc, posteriors)
    model.prior.update_from_statistics(ps, **initial_model.mixture.pi_opts)
    model.likelihood.update_from_statistics(
        ss, **initial_model.mixture.likelihood_opts
    )
    model.delta.update_from_statistics(
        cs, **initial_model.mixture.likelihood_opts
    )
    if dsem is not None:
        sems, _ = get_likelihood_sst(
            initial_model.semantic_delta, dsem, posteriors
        )
        model.semantic_delta.update_from_statistics(
            sems, **initial_model.mixture.likelihood_opts
        )
    return model


def _compute_elbo_delta_impl(model, data):
    """
    Computes the ELBO for the provided data.
    Note: if the data is too large to process, you want to batch around this
          function (see `fit_gmm_step`)

    :param model : Mixture model instance which represents the joint
                   distribution of space, color and the latent z
    :param data : the data points to consider
    :returns elbo: array of elbo's for each data point
    :returns posteriors: array of the posterior distribution q(z) for all data
    """
    d = model.mixture.expand_to_categorical_dims(data)
    logprob = _elbo_logprob_parts(model, d)
    mixdims = tuple(range(-model.mixture.prior.event_dim, 0))
    elbo_contrib = logsumexp(logprob, mixdims)
    elbo = elbo_contrib - _elbo_kl_parts(model, mixdims)
    posteriors = softmax(logprob, mixdims)
    return elbo, posteriors


def _compute_elbo_only_delta_impl(model, data):
    d = model.mixture.expand_to_categorical_dims(data)
    logprob = _elbo_logprob_parts(model, d)
    mixdims = tuple(range(-model.mixture.prior.event_dim, 0))
    elbo_contrib = logsumexp(logprob, mixdims)
    return elbo_contrib - _elbo_kl_parts(model, mixdims)


@jax.jit
def compute_elbo_delta(model, data):
    """
    Computes the ELBO for the provided data.
    Note: if the data is too large to process, you want to batch around this
          function (see `fit_gmm_step`)

    :param model : Mixture model instance which represents the joint
                   distribution of space, color and the latent z
    :param data : the data points to consider
    :returns elbo: array of elbo's for each data point
    :returns posteriors: array of the posterior distribution q(z) for all data
    """
    return _compute_elbo_delta_impl(model, data)


@jax.jit
def compute_elbo_delta_fp32(model, data):
    model = _cast_floating_tree(model, jnp.float32)
    data = data.astype(jnp.float32)
    return _compute_elbo_delta_impl(model, data)


@jax.jit
def compute_elbo_delta_tf32(model, data):
    model = _cast_floating_tree(model, jnp.float32)
    data = data.astype(jnp.float32)
    with jax.default_matmul_precision("tensorfloat32"):
        return _compute_elbo_delta_impl(model, data)


def compute_elbo_delta_with_precision(model, data, precision="fp64"):
    if precision in ("op", "optimized"):
        from vbgs.model.precision_runtime import run_elbo_delta

        return run_elbo_delta(model, data)
    if precision == "tf32":
        return compute_elbo_delta_tf32(model, data)
    if precision == "fp32":
        return compute_elbo_delta_fp32(model, data)
    if precision in ("fp64", "baseline"):
        return compute_elbo_delta(model, data)
    raise ValueError(f"Unsupported VBGS precision mode: {precision}")


@jax.jit
def compute_elbo_only_delta(model, data):
    return _compute_elbo_only_delta_impl(model, data)


@jax.jit
def compute_elbo_only_delta_fp32(model, data):
    model = _cast_floating_tree(model, jnp.float32)
    data = data.astype(jnp.float32)
    return _compute_elbo_only_delta_impl(model, data)


@jax.jit
def compute_elbo_only_delta_tf32(model, data):
    model = _cast_floating_tree(model, jnp.float32)
    data = data.astype(jnp.float32)
    with jax.default_matmul_precision("tensorfloat32"):
        return _compute_elbo_only_delta_impl(model, data)


def compute_elbo_only_delta_with_precision(model, data, precision="fp64"):
    if precision in ("op", "optimized"):
        from vbgs.model.precision_runtime import run_elbo_only_delta

        return run_elbo_only_delta(model, data)
    if precision == "tf32":
        return compute_elbo_only_delta_tf32(model, data)
    if precision == "fp32":
        return compute_elbo_only_delta_fp32(model, data)
    if precision in ("fp64", "baseline"):
        return compute_elbo_only_delta(model, data)
    raise ValueError(f"Unsupported VBGS precision mode: {precision}")


def _gather_components(x, idx):
    flat = x.reshape((x.shape[0], -1))
    gathered = jnp.take(flat, idx, axis=0)
    return gathered.reshape(idx.shape + x.shape[1:])


def _topm_loglik_from_stats(stats, dim, x, idx):
    """Expected log-likelihood of the top-m components from *precomputed* stats.

    ``stats`` is the output of ``dist.expected_posterior_statistics()`` over all
    components. Since ``initial_model`` is fixed for the whole fit, these stats
    are constant, so we compute them once and only gather the ``idx`` subset here
    instead of recomputing the dense (N, 3, 3) inversions/log-dets per batch.
    """
    eta1 = _gather_components(stats.eta.eta_1, idx)
    eta2 = _gather_components(stats.eta.eta_2, idx)
    nu1 = _gather_components(stats.nu.nu_1, idx)
    nu2 = _gather_components(stats.nu.nu_2, idx)

    x = x[:, None, :, :]
    xx = x @ x.mT
    term1 = (x * eta1).sum(axis=(-2, -1))
    term2 = (-0.5 * xx * eta2).sum(axis=(-2, -1))
    term3 = (nu1 + nu2).sum(axis=(-2, -1))
    log_measure = -0.5 * dim * jnp.log(2 * jnp.pi)
    return log_measure + term1 + term2 + term3


def _mvn_expected_log_likelihood_topm(dist, x, idx):
    return _topm_loglik_from_stats(
        dist.expected_posterior_statistics(), dist.dim, x, idx
    )


def _topm_indices(initial_model, xi, top_m):
    x = xi[:, :3, :]
    mean = initial_model.mixture.likelihood.mean
    inv_sigma = initial_model.mixture.likelihood.expected_inv_sigma()
    diff = x[:, None, :, :] - mean[None, :, :, :]
    maha = (diff.mT @ inv_sigma[None, :, :, :] @ diff)[..., 0, 0]
    k = min(int(top_m), int(mean.shape[0]))
    _, idx = jax.lax.top_k(-maha, k)
    return idx


def _candidate_topm_indices(initial_model, xi, candidate_idx, top_m):
    x = xi[:, :3, :]
    mean = initial_model.mixture.likelihood.mean
    inv_sigma = initial_model.mixture.likelihood.expected_inv_sigma()
    cand_mean = _gather_components(mean, candidate_idx)
    cand_inv_sigma = _gather_components(inv_sigma, candidate_idx)
    diff = x[:, None, :, :] - cand_mean
    maha = (diff.mT @ cand_inv_sigma @ diff)[..., 0, 0]
    k = min(int(top_m), int(candidate_idx.shape[1]))
    _, local_idx = jax.lax.top_k(-maha, k)
    return jnp.take_along_axis(candidate_idx, local_idx, axis=1)


def _scatter_counts(idx, values, n_components):
    return jnp.zeros((n_components,), dtype=values.dtype).at[idx].add(values)


def _scatter_mvn_stats(idx, posteriors, x, n_components):
    x_b = x[:, None, :, :]
    xx_b = x_b @ x_b.mT
    eta1_vals = posteriors[..., None, None] * x_b
    eta2_vals = posteriors[..., None, None] * (-0.5 * xx_b)
    counts = posteriors[..., None, None]
    eta1 = jnp.zeros((n_components, x.shape[-2], 1), dtype=x.dtype).at[idx].add(
        eta1_vals
    )
    eta2 = jnp.zeros(
        (n_components, x.shape[-2], x.shape[-2]), dtype=x.dtype
    ).at[idx].add(eta2_vals)
    nu = jnp.zeros((n_components, 1, 1), dtype=x.dtype).at[idx].add(counts)
    return ArrayDict(eta=ArrayDict(eta_1=eta1, eta_2=eta2), nu=ArrayDict(nu_1=nu, nu_2=nu))


def _compute_topm_stats(initial_model, xi, top_m):
    n_components = int(initial_model.mixture.prior.alpha.shape[0])
    n_sem = model_n_semantic(initial_model)
    ds, dc, dsem = split_features(xi, n_sem)
    idx = _topm_indices(initial_model, xi, top_m)

    space_logprob = _mvn_expected_log_likelihood_topm(
        initial_model.mixture.likelihood, ds, idx
    )
    color_logprob = _mvn_expected_log_likelihood_topm(initial_model.delta, dc, idx)
    prior_logprob = jnp.take(initial_model.mixture.prior.log_mean(), idx, axis=0)
    logprob = space_logprob + color_logprob + prior_logprob
    if dsem is not None:
        sem_logprob = _mvn_expected_log_likelihood_topm(
            initial_model.semantic_delta, dsem, idx
        )
        logprob = logprob + sem_logprob

    posteriors = softmax(logprob, axis=-1)
    ps = ArrayDict(eta=ArrayDict(eta_1=_scatter_counts(idx, posteriors, n_components)), nu=None)
    ss = _scatter_mvn_stats(idx, posteriors, ds, n_components)
    cs = _scatter_mvn_stats(idx, posteriors, dc, n_components)
    sems = None
    if dsem is not None:
        sems = _scatter_mvn_stats(idx, posteriors, dsem, n_components)
    return ps, ss, cs, sems, logsumexp(logprob, axis=-1)


compute_topm_stats = jax.jit(_compute_topm_stats, static_argnames=("top_m",))


def _compute_candidate_topm_stats(initial_model, xi, candidate_idx, top_m):
    n_components = int(initial_model.mixture.prior.alpha.shape[0])
    n_sem = model_n_semantic(initial_model)
    ds, dc, dsem = split_features(xi, n_sem)
    idx = _candidate_topm_indices(initial_model, xi, candidate_idx, top_m)

    space_logprob = _mvn_expected_log_likelihood_topm(
        initial_model.mixture.likelihood, ds, idx
    )
    color_logprob = _mvn_expected_log_likelihood_topm(initial_model.delta, dc, idx)
    prior_logprob = jnp.take(initial_model.mixture.prior.log_mean(), idx, axis=0)
    logprob = space_logprob + color_logprob + prior_logprob
    if dsem is not None:
        sem_logprob = _mvn_expected_log_likelihood_topm(
            initial_model.semantic_delta, dsem, idx
        )
        logprob = logprob + sem_logprob

    posteriors = softmax(logprob, axis=-1)
    ps = ArrayDict(
        eta=ArrayDict(eta_1=_scatter_counts(idx, posteriors, n_components)),
        nu=None,
    )
    ss = _scatter_mvn_stats(idx, posteriors, ds, n_components)
    cs = _scatter_mvn_stats(idx, posteriors, dc, n_components)
    sems = None
    if dsem is not None:
        sems = _scatter_mvn_stats(idx, posteriors, dsem, n_components)
    return ps, ss, cs, sems


compute_candidate_topm_stats = jax.jit(
    _compute_candidate_topm_stats, static_argnames=("top_m",)
)


class _TopmCache:
    """Holds the dense, run-constant quantities the sparse E-step needs.

    ``initial_model`` is fixed for the whole fit, so its expected posterior
    statistics, component means and expected inverse covariances never change.
    We materialise them once here and reuse across every batch / frame, turning
    the per-batch cost from O(N) dense inversions into O(points x top_m) gathers.
    """

    def __init__(self, space, color, semantic, prior_log_mean, mean, inv_sigma,
                 space_dim, color_dim, semantic_dim, n_sem):
        self.space = space
        self.color = color
        self.semantic = semantic
        self.prior_log_mean = prior_log_mean
        self.mean = mean
        self.inv_sigma = inv_sigma
        self.space_dim = space_dim
        self.color_dim = color_dim
        self.semantic_dim = semantic_dim
        self.n_sem = n_sem


def _precision_dtype(precision):
    if precision in ("fp32", "tf32"):
        return jnp.float32
    return None


def _maybe_cast(value, dtype):
    if dtype is None or value is None:
        return value
    return _cast_floating_tree(value, dtype)


def build_topm_cache(initial_model, precision="fp64"):
    dtype = _precision_dtype(precision)
    lik = initial_model.mixture.likelihood
    delta = initial_model.delta
    n_sem = model_n_semantic(initial_model)
    semantic = None
    semantic_dim = 0
    if n_sem:
        semantic = initial_model.semantic_delta.expected_posterior_statistics()
        semantic_dim = initial_model.semantic_delta.dim
    return _TopmCache(
        space=_maybe_cast(lik.expected_posterior_statistics(), dtype),
        color=_maybe_cast(delta.expected_posterior_statistics(), dtype),
        semantic=_maybe_cast(semantic, dtype),
        prior_log_mean=_maybe_cast(initial_model.mixture.prior.log_mean(), dtype),
        mean=_maybe_cast(lik.mean, dtype),
        inv_sigma=_maybe_cast(lik.expected_inv_sigma(), dtype),
        space_dim=lik.dim,
        color_dim=delta.dim,
        semantic_dim=semantic_dim,
        n_sem=n_sem,
    )


def _candidate_topm_indices_cached(mean, inv_sigma, x_space, candidate_idx, top_m):
    cand_mean = _gather_components(mean, candidate_idx)
    cand_inv_sigma = _gather_components(inv_sigma, candidate_idx)
    diff = x_space[:, None, :, :] - cand_mean
    maha = (diff.mT @ cand_inv_sigma @ diff)[..., 0, 0]
    k = min(int(top_m), int(candidate_idx.shape[1]))
    _, local_idx = jax.lax.top_k(-maha, k)
    return jnp.take_along_axis(candidate_idx, local_idx, axis=1)


def _compute_candidate_topm_stats_cached(
    space, color, semantic, prior_log_mean, mean, inv_sigma,
    xi, candidate_idx, top_m, space_dim, color_dim, semantic_dim, n_sem
):
    n_components = int(prior_log_mean.shape[0])
    ds, dc, dsem = split_features(xi, n_sem)
    idx = _candidate_topm_indices_cached(mean, inv_sigma, ds, candidate_idx, top_m)

    space_logprob = _topm_loglik_from_stats(space, space_dim, ds, idx)
    color_logprob = _topm_loglik_from_stats(color, color_dim, dc, idx)
    prior_logprob = jnp.take(prior_log_mean, idx, axis=0)
    logprob = space_logprob + color_logprob + prior_logprob
    if dsem is not None:
        sem_logprob = _topm_loglik_from_stats(semantic, semantic_dim, dsem, idx)
        logprob = logprob + sem_logprob

    posteriors = softmax(logprob, axis=-1)
    ps = ArrayDict(
        eta=ArrayDict(eta_1=_scatter_counts(idx, posteriors, n_components)),
        nu=None,
    )
    ss = _scatter_mvn_stats(idx, posteriors, ds, n_components)
    cs = _scatter_mvn_stats(idx, posteriors, dc, n_components)
    sems = None
    if dsem is not None:
        sems = _scatter_mvn_stats(idx, posteriors, dsem, n_components)
    return ps, ss, cs, sems, logsumexp(logprob, axis=-1)


compute_candidate_topm_stats_cached = jax.jit(
    _compute_candidate_topm_stats_cached,
    static_argnames=("top_m", "space_dim", "color_dim", "semantic_dim", "n_sem"),
)


def _compute_candidate_topm_elbo_cached(
    space,
    color,
    semantic,
    prior_log_mean,
    mean,
    inv_sigma,
    xi,
    candidate_idx,
    top_m,
    space_dim,
    color_dim,
    semantic_dim,
    n_sem,
):
    ds, dc, dsem = split_features(xi, n_sem)
    idx = _candidate_topm_indices_cached(mean, inv_sigma, ds, candidate_idx, top_m)
    logprob = (
        _topm_loglik_from_stats(space, space_dim, ds, idx)
        + _topm_loglik_from_stats(color, color_dim, dc, idx)
        + jnp.take(prior_log_mean, idx, axis=0)
    )
    if dsem is not None:
        logprob = logprob + _topm_loglik_from_stats(semantic, semantic_dim, dsem, idx)
    return logsumexp(logprob, axis=-1)


compute_candidate_topm_elbo_cached = jax.jit(
    _compute_candidate_topm_elbo_cached,
    static_argnames=("top_m", "space_dim", "color_dim", "semantic_dim", "n_sem"),
)


def fit_gmm_step(
    initial_model,
    model,
    data,
    batch_size,
    prior_stats=None,
    space_stats=None,
    color_stats=None,
    semantic_stats=None,
    precision="fp64",
    top_m=None,
    candidate_indices=None,
    topm_cache=None,
    low_elbo_count=0,
):
    """
    Compute a single update step for the `DeltaMixture` using the assignments
    upon the initial model, but adding the sst to the model.

    :param initial_model: DeltaMixture before having applied a single update
    :param model: DeltaMixture of the model having applied the previous updates
                  is used to apply the update upon
    :param data: The data to fit the model to. Preferably a numpy array, to
                 only populate the GPU when it's necessary.
    :param batch_size: size of a single batch processed by GPU
    :param prior_stats: The collected sufficient statistics of the prior. None
                        at step 0, after that it should contain the prior_stats
                        returned at the previous step.
    :param space_stats: The sufficient statistics of the spatial likelihood.
                        None at step 0, after that it should contain the
                        space_stats returned at the previous step.
    :param color_stats: The sufficient statistics of the color likelihood. None
                        at step 0, after that it should contain the color_stats
                        returned at the previous step.
    :returns model: DeltaMixture model after updating
    """
    dtype = _precision_dtype(precision)
    if dtype is not None:
        initial_model = _cast_floating_tree(initial_model, dtype)
        model = _cast_floating_tree(model, dtype)
        prior_stats = _maybe_cast(prior_stats, dtype)
        space_stats = _maybe_cast(space_stats, dtype)
        color_stats = _maybe_cast(color_stats, dtype)
        semantic_stats = _maybe_cast(semantic_stats, dtype)

    # The sparse E-step reads only run-constant quantities of initial_model, so
    # build them once here instead of recomputing dense (N,3,3) stats per batch.
    if top_m is not None and candidate_indices is not None and topm_cache is None:
        topm_cache = build_topm_cache(initial_model, precision=precision)

    low_elbo_count = int(low_elbo_count or 0)
    collect_low_elbo = low_elbo_count > 0
    low_elbo_idcs = []
    low_elbo_values = []
    n_batches = int(np.ceil(data.shape[0] / batch_size))
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        xi = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        xi = jnp.expand_dims(jnp.array(xi), -1)
        if dtype is not None:
            xi = xi.astype(dtype)
        candidate_idx = None
        if candidate_indices is not None:
            candidate_idx = jnp.array(
                candidate_indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            )

        size = xi.shape[0]
        elbo = None
        if top_m is not None and candidate_idx is not None:
            ps, ss, cs, sems, elbo = compute_candidate_topm_stats_cached(
                topm_cache.space,
                topm_cache.color,
                topm_cache.semantic,
                topm_cache.prior_log_mean,
                topm_cache.mean,
                topm_cache.inv_sigma,
                xi[:size],
                candidate_idx[:size],
                int(top_m),
                int(topm_cache.space_dim),
                int(topm_cache.color_dim),
                int(topm_cache.semantic_dim),
                int(topm_cache.n_sem),
            )
        elif top_m is not None:
            ps, ss, cs, sems, elbo = compute_topm_stats(
                initial_model, xi[:size], int(top_m)
            )
        else:
            if size < batch_size:
                # Concat zeros, so that the posteriors are still computed with jit
                # NOTE: the elbo will not be correct since it's contributing the
                # log likelihood of the augmented zeros. But since we don't use the
                # elbo here, it is not a problem.
                xi = jnp.concatenate(
                    [xi, jnp.zeros((batch_size - size, *xi.shape[1:]))],
                    axis=0,
                )
                elbo, posteriors = compute_elbo_delta_with_precision(
                    initial_model, xi, precision
                )
                xi = xi[:size]
                elbo = elbo[:size]
                posteriors = posteriors[:size]
            else:
                elbo, posteriors = compute_elbo_delta_with_precision(
                    initial_model, xi, precision
                )

            cat_i = initial_model.mixture.expand_to_categorical_dims(xi)
            n_sem = model_n_semantic(initial_model)
            ds, dc, dsem = split_features(cat_i, n_sem)

            ps = model.mixture._to_stats(
                posteriors, initial_model.mixture.get_sample_dims(cat_i)
            )
            ss, _ = get_likelihood_sst(
                initial_model.mixture.likelihood, ds, posteriors
            )
            cs, _ = get_likelihood_sst(initial_model.delta, dc, posteriors)
            sems = None
            if dsem is not None:
                sems, _ = get_likelihood_sst(
                    initial_model.semantic_delta, dsem, posteriors
                )

        if collect_low_elbo and elbo is not None:
            batch_elbo = np.asarray(jax.block_until_ready(elbo))[:size]
            keep = min(low_elbo_count, int(batch_elbo.shape[0]))
            if keep > 0:
                local_idcs = np.argpartition(batch_elbo, keep - 1)[:keep]
                low_elbo_idcs.append(local_idcs + batch_start)
                low_elbo_values.append(batch_elbo[local_idcs])

        if batch_idx == 0 and prior_stats is None:
            prior_stats = ps
            space_stats = ss
            color_stats = cs
            semantic_stats = sems
        else:
            prior_stats = utils.apply_add(ps, prior_stats)
            space_stats = utils.apply_add(ss, space_stats)
            color_stats = utils.apply_add(cs, color_stats)
            if sems is not None:
                semantic_stats = utils.apply_add(sems, semantic_stats)

    model.mixture.prior.update_from_statistics(
        prior_stats, **initial_model.mixture.pi_opts
    )
    model.mixture.likelihood.update_from_statistics(
        space_stats, **initial_model.mixture.likelihood_opts
    )
    model.delta.update_from_statistics(
        color_stats, **initial_model.mixture.likelihood_opts
    )
    if semantic_stats is not None:
        model.semantic_delta.update_from_statistics(
            semantic_stats, **initial_model.mixture.likelihood_opts
        )

    if not collect_low_elbo:
        return model, prior_stats, space_stats, color_stats, semantic_stats

    if low_elbo_idcs:
        all_idcs = np.concatenate(low_elbo_idcs)
        all_values = np.concatenate(low_elbo_values)
        keep = min(low_elbo_count, int(all_values.shape[0]))
        selected = np.argpartition(all_values, keep - 1)[:keep]
        low_elbo = {
            "point_indices": all_idcs[selected],
            "elbo": all_values[selected],
        }
    else:
        low_elbo = {
            "point_indices": np.asarray([], dtype=np.int64),
            "elbo": np.asarray([], dtype=np.float32),
        }
    return model, prior_stats, space_stats, color_stats, semantic_stats, low_elbo
