# -*- coding: utf-8 -*-

# Copyright 2022 yukoga. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import jax.numpy as jnp
import jax.random as random
from jax_cltv.dists.base import BaseContinuousDist
from jax.scipy.stats import norm


class Normal(BaseContinuousDist):
    def __init__(
        self, loc: jnp.DeviceArray = 0.0, scale: jnp.DeviceArray = 1.0
    ) -> BaseContinuousDist:
        self.loc = loc
        self.scale = scale

    def logpdf(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        """Calc log-pdf of normal distribution for given data.
        and probability distribution in log pdf form.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.

        Returns
        -------
        log-likelihood at data x: jnp.DeviceArray with a scalar value.
        """
        return norm.logpdf(x, loc=self.loc, scale=self.scale)

    def sample(self, rng_key: jnp.DeviceArray, size: int) -> jnp.DeviceArray:
        """Generate random values from a normal distribution.

        Parameters
        ----------
        rng_key: int or array_like
            random key in the form of PRNG key or integer.
        size: int or tuple of shape.
            the number of random variables.

        Returns
        -------
        samples: sampled random values in the form of jnp.DeviceArray.
        """
        if not isinstance(rng_key, jnp.DeviceArray):
            rng_key = random.PRNGKey(rng_key)

        return self.loc + self.scale * random.normal(rng_key, (size,))


def loglikelihood(
    x: jnp.DeviceArray, loc: jnp.DeviceArray, scale: jnp.DeviceArray
) -> tuple:
    """Calc log-likelihood of the normal distribution for given data.

    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    loc: array_like
        location parameter in the form of (D, N) array where N is sample size
        and D is dimension size.
    scale: array_like
        scale parameter in the form of (N,) array where N is sample size.

    Returns
    -------
    (loglikelihood, Normal): tuple
        log-likelihood for given data x: jnp.DeviceArray with a scalar value
        and
        the instance of Normal distribution for given parameters.
    """
    d = Normal(loc, scale)
    return d.loglikelihood(x), d


def neg_loglikelihood(
    x: jnp.DeviceArray, loc: jnp.DeviceArray, scale: jnp.DeviceArray
) -> tuple:
    """Calc negative log-likelihood of the normal distribution for given data.

    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    loc: array_like
        location parameter in the form of (D, N) array where N is sample size
        and D is dimension size.
    scale: array_like
        scale parameter in the form of (N,) array where N is sample size.

    Returns
    -------
    (neg_loglikelihood, Normal): tuple
        negative log-likelihood for given data x: jnp.DeviceArray
        with a scalar value and
        the instance of Normal distribution for given parameters.
    """
    d = Normal(loc, scale)
    return d.negloglikelihood(x), d


def rv_samples(
    loc: jnp.DeviceArray = 0.0,
    scale: jnp.DeviceArray = 1.0,
    rng_key: jnp.DeviceArray = 1,
    size=100,
) -> tuple:
    """Generate random values from a normal distribution.

    Parameters
    ----------
    loc: array_like
        location parameter in the form of (D, N) array where N is sample size
        and D is dimension size.
    scale: array_like
        scale parameter in the form of (N,) array where N is sample size.
    rng_key: int or array_like
        random key in the form of PRNG key or integer.
    size: int or tuple of shape.
        the number of random variables.

    Returns
    -------
    (samples, Normal): tuple
        random values sampleing from Normal distribution specified
        by given parameters and
        the instance of Normal distribution for given parameters.
    """
    d = Normal(loc, scale)
    return d.sample(rng_key, size), d
