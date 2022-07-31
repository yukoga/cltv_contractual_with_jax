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


N = 100 # number of samples.
mu_true = jnp.array([2., 2.]) # True parameter
sigma_true = 1. # True variance


def toy_data(N, theta, rv_key):
    w = theta[:-1]
    D = w.shape[0]
    sigma = theta[-1] 
    _x = random.normal(rv_key, (N, D))
    _y = jnp.dot(_x, w) + sigma * random.normal(rv_key, (N,))
    return _x, _y


@pytest.fixture(scope='session')
def data():
    rv_key = random.PRNGKey(1)
    mu = mu_true 
    sigma = sigma_true
    theta = jnp.append(mu, sigma)
    _x, _y = toy_data(N, theta, rv_key)
    loglik = jnp.sum(jstats.norm.logpdf(_y, loc=jnp.dot(_x, mu), scale=sigma))
    return rv_key, theta, loglik, (_x, _y) 