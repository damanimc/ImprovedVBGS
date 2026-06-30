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

from functools import partial

import equinox

import jax
import jax.numpy as jnp

from vbgs.vi.conjugate.mvn import MultivariateNormal
from vbgs.vi.models.mixture import Mixture

from vbgs.model.feature_layout import N_COLOR, N_SPATIAL, model_n_semantic
from vbgs.model.utils import transform_mvn


class DeltaMixture(equinox.Module):
    """
    A small compositional class to allow for the use of previously written code
    that calls `model.likelihood` and `model.prior`
    """

    mixture: Mixture
    delta: MultivariateNormal
    semantic_delta: MultivariateNormal | None = None

    def __init__(self, mixture, delta, semantic_delta=None):
        self.mixture = mixture
        self.delta = delta
        self.semantic_delta = semantic_delta

    @property
    def likelihood(self):
        return self.mixture.likelihood

    @property
    def prior(self):
        return self.mixture.prior

    def denormalize(self, params, clip_val=None):
        """
        Invert the normalization step applied to the data, such that
        the model is now in the space of the original data.
        """
        mu_uv = self.mixture.likelihood.mean[:, :, 0]
        si_uv = self.mixture.likelihood.expected_sigma()

        mu_rgb = self.delta.likelihood.mean[:, :, 0]
        si_rgb = jnp.eye(N_COLOR).reshape(-1, N_COLOR, N_COLOR)

        n_sem = model_n_semantic(self)
        if n_sem > 0:
            mu_sem = self.semantic_delta.likelihood.mean[:, :, 0]
            si_sem = jnp.eye(n_sem).reshape(-1, n_sem, n_sem)
            n = N_SPATIAL + N_COLOR + n_sem
        else:
            mu_sem = None
            n = N_SPATIAL + N_COLOR

        mu = jnp.zeros((mu_uv.shape[0], n))
        mu = mu.at[:, :N_SPATIAL].set(mu_uv)
        mu = mu.at[:, N_SPATIAL : N_SPATIAL + N_COLOR].set(mu_rgb)
        if mu_sem is not None:
            mu = mu.at[:, N_SPATIAL + N_COLOR :].set(mu_sem)

        si = jnp.zeros((mu_uv.shape[0], n, n))
        si = si.at[:, :N_SPATIAL, :N_SPATIAL].set(si_uv)
        si = si.at[
            :, N_SPATIAL : N_SPATIAL + N_COLOR, N_SPATIAL : N_SPATIAL + N_COLOR
        ].set(si_rgb)
        if mu_sem is not None:
            si = si.at[:, N_SPATIAL + N_COLOR :, N_SPATIAL + N_COLOR :].set(si_sem)

        mu, si = jax.vmap(
            partial(
                transform_mvn,
                params["stdevs"].flatten(),
                params["offset"].flatten(),
            )
        )(mu, si)

        if clip_val is not None:
            si_diag = jnp.diagonal(si, axis1=1, axis2=2).clip(
                clip_val, jnp.inf
            )
            si = jax.vmap(lambda x, y: jnp.fill_diagonal(x, y, inplace=False))(
                si, si_diag
            )

        return mu, si
