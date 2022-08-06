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


import pytest
import jax.numpy as jnp
from jax import random
from jax.scipy import stats as jstats
from scipy.stats import geom


N = 100 # number of samples.
params = {
    'mu': 1.,       # location parameter for normal distribution.
    'sigma': 2.,    # scale parameter for normal distribution.
    'alpha': 2.,    # First shape parameter for beta distribution.
    'beta': 1.,     # Second shape parameter for beta distribution.
    'theta': .5     # Probability for specific event happen after some trials.
}


def toy_data(N, params, rv_key):
    mu = params['mu']
    sigma = params['sigma']
    alpha = params['alpha']
    beta = params['beta']
    theta = params['theta']

    dist = dict()
    dist['key'] = rv_key
    dist['normal'] = { 'mu': mu, 'sigma': sigma }
    _x = random.normal(rv_key, (N,))
    _y = mu + sigma * random.normal(rv_key, (N,))
    dist['normal']['rv'] = (_x, _y)
    dist['normal']['pdf'] = jstats.norm.pdf(_y, loc=mu, scale=sigma)
    dist['normal']['loglik'] = jnp.sum(jstats.norm.logpdf(_y,
            loc=mu, scale=sigma))

    _y = random.beta(rv_key, alpha, beta, (N,))
    dist['beta'] = {
        'alpha': alpha, 'beta': beta,
        'rv': _y,
        'loglik': jnp.sum(jstats.beta.logpdf(_y, alpha, beta))
    }

    # _y = random.geom(rv_key, theta, (N,))
    dist['geom'] = {
        'theta': theta,
        # 'rv': _y,
        # 'loglik': jnp.sum(jstats.geom.logpmf(_y, theta))
    }

    del _x, _y
    return dist


@pytest.fixture(scope='session')
def data():
    rv_key = random.PRNGKey(1)
    dist = toy_data(N, params, rv_key)
    return dist