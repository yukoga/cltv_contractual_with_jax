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


from jax_cltv.dists.normal import (Normal, loglikelihood, neg_loglikelihood) 


def test_instantiate(data):
    d = Normal(.5, 1.)
    assert 'Normal' == d.__class__.__name__, 'It should be an instance of Normal, '
    f'but {d.__class__.__name__}.'


def test_loglikelihood(data):
    mu = data['normal']['mu']
    sigma = data['normal']['sigma']
    _x, _y = data['normal']['rv']
    loglik_true = data['normal']['loglik']
    loglik, d = loglikelihood(_y, mu, sigma) 

    assert 'Normal' == d.__class__.__name__, 'It should be an instance of Normal, '
    f'but {d.__class__.__name__}.'
    assert loglik_true == loglik, 'loglik is wrong. '
    f'{loglik_true} is expected, but {loglik} is.'


def test_neg_loglikelihood(data):
    mu = data['normal']['mu']
    sigma = data['normal']['sigma']
    _x, _y = data['normal']['rv']
    neg_loglik, _ = neg_loglikelihood(_y, mu, sigma) 
    neg_loglik_true = (-1. * data['normal']['loglik']) / _x.shape[0]

    assert neg_loglik == neg_loglik_true, 'neg_loglik is wrong. '
    f' {neg_loglik_true} is expected, but {neg_loglik} is.'


def test_sample(data):
    rv_key = data['key']
    mu = data['normal']['mu']
    sigma = data['normal']['sigma']
    _x, _y = data['normal']['rv']

    nd = Normal(mu, sigma)
    sample = nd.sample(rv_key, size=10000)
    assert round(sample.mean()) == mu, f'Mean of standard normal distribution should be close to {mu}.'
    f'but {round(sample.mean())}.'
    assert round(sample.std()) == sigma, f'Variance of standard normal distribution should be close to {sigma}.'
    f'but {round(sample.std())}.'
