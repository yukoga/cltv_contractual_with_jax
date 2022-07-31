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


# from typing import Callable
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from jax.scipy.stats import beta as jbeta


def normal_log_likelihood(x: jnp.DeviceArray,
                          loc: jnp.DeviceArray,
                          scale: jnp.DeviceArray,
                          **kwargs) -> jnp.DeviceArray:
    """ Calc log likelihood of normal distribution for given data.
    and probability distribution in log pdf form.
    
    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    loc: array_like
        location parameter for normal distribution.
    scale: matrix_like
        scale parameter for normal distribution.
    kwargs: dict
        any other parameters for probabilistic distributions.

    Returns
    -------
    log-likelihood at data x: jnp.DeviceArray with dtype float.
    """
    log_lik = jnorm.logpdf(x, loc, scale) 
    return jnp.sum(log_lik)


def beta_log_likelihood(x: jnp.DeviceArray,
                        alpha: jnp.DeviceArray,
                        beta: jnp.DeviceArray,
                        **kwargs: dict) -> jnp.DeviceArray:
    """ Calc log likelihood of beta distribution for given data.
    and probability distribution in log pdf form.
    
    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    alpha: array_like
        alpha parameter for beta distribution.
    beta: array_like
        scale parameter for beta distribution.
    kwargs: dict
        any other parameters for probabilistic distributions.

    Returns
    -------
    log-likelihood at data x: jnp.DeviceArray with dtype float.
    """
    log_lik = jbeta.logpdf(x, alpha, beta)
    return jnp.sum(log_lik)


def neg_log_likelihood(theta: jnp.DeviceArray,
                       data: tuple,
                       dist: str,
                       **kwargs: dict) -> jnp.DeviceArray:
    """ Calc log negative likelihood for given parameters, data
    and probability distribution in log pdf form.
    
    Parameters
    ----------
    theta: array_like
        parameters for given probability distribution.
        theta[:-1] must be location parameters and theta[-1] to be scale.
    data: tuple
        data in the form of (features, target)
    dist: str
        Type of probability distribution. Only 'normal' and 'beta' are supported.
    kwargs: dict
        any other parameters for probabilistic distributions.
    Returns
    -------
    negative log-likelihood at data: jnp.DeviceArray with dtype float.
    """
    x, y = data
    lik = None
    N = x.shape[0]
    if dist == 'normal':
        loc = jnp.dot(x, theta[:-1])
        scale = theta[-1]
        lik = normal_log_likelihood(y, loc, scale)
    elif dist == 'beta':
        alpha = theta[0]
        beta = theta[1]
        lik = beta_log_likelihood(y, alpha, beta)
    else:
        pass
    return (-1. * lik) / N