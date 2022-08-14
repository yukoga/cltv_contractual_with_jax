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
from abc import ABC, abstractmethod


class BaseDist(ABC):
    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def sample(self):
        pass


class BaseDiscreteDist(BaseDist, ABC):
    @abstractmethod
    def logpmf(self):
        """
        Abstract method for log-probability density function.
        """
        pass

    def pmf(self, x, **kwargs):
        """
        Calculate probability density function.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.
        kwargs: dict
            any other parameters for probabilistic distributions.

        Returns
        -------
        Probability density function values for given data x: jnp.DeviceArray.
        """
        return jnp.exp(self.logpmf(x, **kwargs))

    def loglikelihood(self, x, **kwargs):
        """Calc log-likelihood of the distribution for given data.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.
        kwargs: dict
            any other parameters for probabilistic distributions.

        Returns
        -------
        log-likelihood for given data x: jnp.DeviceArray with a scalar value.
        """
        return jnp.sum(self.logpmf(x, **kwargs))

    def negloglikelihood(self, x, **kwargs):
        """Calc negative log-likelihood of the distribution for given data.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.
        kwargs: dict
            any other parameters for probabilistic distributions.

        Returns
        -------
        negative log-likelihood for given data x: jnp.DeviceArray with a scalar value.
        """
        sample_size = x.shape[0]
        return (-1.0 * self.loglikelihood(x, **kwargs)) / sample_size


class BaseContinuousDist(BaseDist, ABC):
    @abstractmethod
    def logpdf(self):
        """
        Abstract method for log-probability density function.
        """
        pass

    def pdf(self, x, **kwargs):
        """
        Calculate probability density function.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.
        kwargs: dict
            any other parameters for probabilistic distributions.

        Returns
        -------
        Probability density function values for given data x: jnp.DeviceArray.
        """
        return jnp.exp(self.logpdf(x, **kwargs))

    def loglikelihood(self, x, **kwargs):
        """Calc log-likelihood of the distribution for given data.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.
        kwargs: dict
            any other parameters for probabilistic distributions.

        Returns
        -------
        log-likelihood for given data x: jnp.DeviceArray with a scalar value.
        """
        return jnp.sum(self.logpdf(x, **kwargs))

    def negloglikelihood(self, x, **kwargs):
        """Calc negative log-likelihood of the distribution for given data.

        Parameters
        ----------
        x: array_like
            observed data in the form of 1-D vector.
        kwargs: dict
            any other parameters for probabilistic distributions.

        Returns
        -------
        negative log-likelihood for given data x: jnp.DeviceArray with a scalar value.
        """
        sample_size = x.shape[0]
        return (-1.0 * self.loglikelihood(x, **kwargs)) / sample_size
