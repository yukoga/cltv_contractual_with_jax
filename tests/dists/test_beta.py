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

from jax_cltv.dists.beta import (
    Beta,
    loglikelihood,
    neg_loglikelihood,
    rv_samples,
)


def test_instantiate(data):
    d = Beta(0.5, 1.0)
    assert "Beta" == d.__class__.__name__, "It should be an instance of Beta, "
    f"but {d.__class__.__name__}."


def test_get_params(data):
    d = Beta(0.5, 1.0)
    params = d.params

    assert set(params.keys()) == {
        "a",
        "b",
    }, "The instance of Beta distribution should have "
    f"parameters {'a', 'b'}, but {params.keys()}. "

    assert (
        params["a"] == 0.5
    ), f"The parameter a should be 0.5, but {params['a']}."
    assert (
        params["b"] == 1.0
    ), f"The parameter b should be 1.0, but {params['b']}."


def test_set_params(data):
    d = Beta(0.5, 1.0)
    d.params = {"a": 2.0, "b": 3.0}
    params = d.params

    assert set(params.keys()) == {
        "a",
        "b",
    }, "The instance of Beta distribution should have "
    f"parameters {'a', 'b'}, but {params.keys()}. "

    assert (
        params["a"] == 2.0
    ), f"The parameter a should be 2.0, but {params['a']}."
    assert (
        params["b"] == 3.0
    ), f"The parameter b should be 3.0, but {params['b']}."


def test_beta_pdf(data):
    alpha = data["beta"]["alpha"]
    beta = data["beta"]["beta"]
    _y = data["beta"]["rv"]
    pdf_true = data["beta"]["pdf"]
    d = Beta(alpha, beta)
    pdf = d.pdf(_y)

    assert (pdf_true == pdf).all(), "pdf is wrong. "
    f"{pdf_true} is expected, but {pdf} is."


def test_loglikelihood(data):
    alpha = data["beta"]["alpha"]
    beta = data["beta"]["beta"]
    _y = data["beta"]["rv"]
    loglik_true = data["beta"]["loglik"]
    loglik, d = loglikelihood(_y, alpha, beta)

    assert (
        "Beta" == d.__class__.__name__
    ), "It should be an instance of Normal, "
    f"but {d.__class__.__name__}."
    assert loglik_true == loglik, "loglik is wrong. "
    f"{loglik_true} is expected, but {loglik} is."


def test_neg_loglikelihood(data):
    alpha = data["beta"]["alpha"]
    beta = data["beta"]["beta"]
    _y = data["beta"]["rv"]
    neg_loglik, _ = neg_loglikelihood(_y, alpha, beta)
    neg_loglik_true = (-1.0 * data["beta"]["loglik"]) / _y.shape[0]

    assert neg_loglik == neg_loglik_true, "neg_loglik is wrong. "
    f" {neg_loglik_true} is expected, but {neg_loglik} is."


def test_rv_samples(data):
    rv_key = data["key"]
    alpha = data["beta"]["alpha"]
    beta = data["beta"]["beta"]
    _y = data["beta"]["rv"]

    mu = alpha / (alpha + beta)
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

    samples, _ = rv_samples(alpha, beta, rv_key, _y.shape[0])
    assert (
        round((samples.mean() - mu) / mu, 2) < 0.1
    ), "Mean of beta distribution should be "
    f"close to {mu}, but {round(samples.mean(), 2)}."

    assert (
        round((samples.var() - var) / var, 2) < 0.1
    ), "Variance of beta distribution should be "
    f"close to {round(var, 2)}, but {round(samples.var(), 2)}."
