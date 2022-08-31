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
from jax_cltv.dists.base import BaseDiscreteDist
from jax.scipy.stats import geom


class Geometric(BaseDiscreteDist):
    def __init__(self, theta: jnp.DeviceArray = 0.5) -> BaseDiscreteDist:
        self.theta = theta
        self.__params = {"theta": theta}

    @property
    def params(self) -> dict:
        """Return parameters which characterize the distribution.

        Returns:
            dict: parameters of Geometric distribution.
        """
        return self.__params

    @params.setter
    def params(self, params):
        self.__params = params

    def logpmf(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        """Calc log-pdf of geometric distribution for given data.
        and probability distribution in log pdf form.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.
        theta: array_like
            Shape parameter of a geometric distribution.

        Returns
        -------
        Log of the probability mass function at data x: jnp.DeviceArray.
        """
        return geom.logpmf(x, self.theta)

    def sample(self, rng_key: jnp.DeviceArray, size: int) -> jnp.DeviceArray:
        """Generate random values from a geometric distribution.

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

        return jnp.ceil(
            jnp.log(random.uniform(rng_key, (size,))) / jnp.log1p(-self.theta)
        )

    def logsf(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        k = jnp.floor(x)
        return k * jnp.log1p(-self.theta)

    def sf(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        return jnp.exp(self.logsf(x))


def loglikelihood(x: jnp.DeviceArray, theta: jnp.DeviceArray) -> tuple:
    """Calc log-likelihood of the geometric distribution for given data.

    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    theta: array_like
        Shape parameter of a geometric distribution.

    Returns
    -------
    (loglikelihood, Geometric): tuple
        Log of the probability mass function at data x: jnp.DeviceArray
        and
        the instance of Geometric distribution for given parameters.
    """
    d = Geometric(theta)
    return d.loglikelihood(x), d


def neg_loglikelihood(x: jnp.DeviceArray, theta: jnp.DeviceArray) -> tuple:
    """Calc negative log-likelihood of the geometric distribution
    for given data.

    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    theta: array_like
        Shape parameter of a geometric distribution.

    Returns
    -------
    (neg_loglikelihood, Geometric): tuple
        Negative sum of Log-probability mass function
        at data x: jnp.DeviceArray and
        the instance of Geometric distribution for given parameters.
    """
    d = Geometric(theta)
    return d.negloglikelihood(x), d


def rv_samples(
    theta: jnp.DeviceArray = 0.5, rng_key: jnp.DeviceArray = 1, size=100
) -> tuple:
    """Generate random values from a geometric distribution.

    Parameters
    ----------
    theta: array_like
        Shape parameter of a geometric distribution.
    rng_key: int or array_like
        random key in the form of PRNG key or integer.
    size: int or tuple of shape.
        the number of random variables.

    Returns
    -------
    (samples, Geometric): tuple
        random values sampleing from Geometric distribution specified
        by given parameters and
        the instance of Geometric distribution for given parameters.
    """
    d = Geometric(theta)
    return d.sample(rng_key, size), d


def survival_functions(x: jnp.DeviceArray, theta: jnp.DeviceArray) -> tuple:
    d = Geometric(theta)
    return d.sf(x), d
