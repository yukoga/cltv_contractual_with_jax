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
from jax_cltv.model import neg_log_likelihood
from jax_cltv.model import normal_log_likelihood, beta_log_likelihood, geom_log_likelihood


def test_geom_log_likelihood(data):
    rv_key, theta, loglik_true_, (x, y), dist = data
    p = theta[2] * .5
    loglik_true = jnp.round(jnp.sum(jstats.geom.logpmf(y, p)), 2)
    loglik = jnp.round(geom_log_likelihood(y, p), 2)

    assert loglik.dtype == 'float32', 'geom_log_likelihood type must be float32, '
    f'but {loglik.dtype}'
    assert loglik == loglik_true, 'geom_log_likelihood for test toy data must be '
    f'{loglik_true}, but {loglik}'


def test_geom_neg_log_likelihood(data):
    rv_key, theta, loglik_true_, (x, y), dist = data
    N = x.shape[0]
    p = theta[2] * .5

    loglik_true = jnp.round(jnp.sum(jstats.geom.logpmf(y, p)), 2)
    neg_lik_true = jnp.round((-1. * loglik_true) / N, 2)
    neg_lik = jnp.round(neg_log_likelihood(theta, (x, y), 'geometric'), 2)

    assert neg_lik.dtype == 'float32', 'neg_log_likelihood type must be float32, '
    f'but {neg_lik.dtype}'
    assert neg_lik == neg_lik_true, 'Negative log-likelihood for test toy data must be '
    f'{neg_lik_true}, but {neg_lik}'