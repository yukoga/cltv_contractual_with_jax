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
from jax.scipy.stats import beta


class Beta(BaseContinuousDist):
    def __init__(
        self, a: jnp.DeviceArray = 1.0, b: jnp.DeviceArray = 1.0
    ) -> BaseContinuousDist:
        self.a = a
        self.b = b

    def logpdf(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        """Calc log-pdf of beta distribution for given data.
        and probability distribution in log pdf form.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.

        Returns
        -------
        log-likelihood at data x: jnp.DeviceArray with a scalar value.
        """
        return beta.logpdf(x, a=self.a, b=self.b)

    def sample(self, rng_key: jnp.DeviceArray, size: int) -> jnp.DeviceArray:
        """Generate random values from a beta distribution.

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

        return random.beta(rng_key, self.a, self.b, (size,))


def loglikelihood(
    x: jnp.DeviceArray, a: jnp.DeviceArray, b: jnp.DeviceArray
) -> tuple:
    """Calc log-likelihood of the beta distribution for given data.

    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    a: array_like
        The first shape parameter in the form of (D, N) array
        where N is sample size and D is dimension size.
    b: array_like
        2nd shape parameter in the form of (N,) array where N is sample size.

    Returns
    -------
    (loglikelihood, Beta): tuple
        log-likelihood for given data x: jnp.DeviceArray with shape parameters
        and
        the instance of Beta distribution for given parameters.
    """
    d = Beta(a, b)
    return d.loglikelihood(x), d


def neg_loglikelihood(
    x: jnp.DeviceArray, a: jnp.DeviceArray, b: jnp.DeviceArray
) -> tuple:
    """Calc negative log-likelihood of the beta distribution for given data.

    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    a: array_like
        The first shape parameter in the form of (D, N) array
        where N is sample size and D is dimension size.
    b: array_like
        2nd shape parameter in the form of (N,) array where N is sample size.

    Returns
    -------
    (neg_loglikelihood, Beta): tuple
        negative log-likelihood for given data x: jnp.DeviceArray
        with shape parameters and
        the instance of Beta distribution for given parameters.
    """
    d = Beta(a, b)
    return d.negloglikelihood(x), d


def rv_samples(
    a: jnp.DeviceArray = 1.0,
    b: jnp.DeviceArray = 1.0,
    rng_key: jnp.DeviceArray = 1,
    size=100,
) -> tuple:
    """Generate random values from a beta distribution.

    Parameters
    ----------
    a: array_like
        The first shape parameter in the form of (D, N) array
        where N is sample size and D is dimension size.
    b: array_like
        2nd shape parameter in the form of (N,) array where N is sample size.
    rng_key: int or array_like
        random key in the form of PRNG key or integer.
    size: int or tuple of shape.
        the number of random variables.

    Returns
    -------
    (samples, Beta): tuple
        random values sampleing from Beta distribution specified
        by given parameters and
        the instance of Beta distribution for given parameters.
    """
    d = Beta(a, b)
    return d.sample(rng_key, size), d
