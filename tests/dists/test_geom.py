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

import numpy as np
import jax.numpy as jnp
from jax_cltv.dists.geom import (
    Geometric,
    loglikelihood,
    neg_loglikelihood,
    rv_samples,
    survival_functions,
)


def test_instantiate(data):
    d = Geometric(0.5)
    assert (
        "Geometric" == d.__class__.__name__
    ), "It should be an instance of Geometric, "
    f"but {d.__class__.__name__}."


def test_get_params(data):
    d = Geometric(0.5)
    params = d.params

    assert set(params.keys()) == {
        "theta",
    }, "The instance of Geometric distribution should have "
    f"parameters {'theta'}, but {params.keys()}. "

    assert (
        params["theta"] == 0.5
    ), f"The parameter theta should be 0.5, but {params['theta']}."


def test_set_params(data):
    d = Geometric(0.5)
    d.params = {"theta": 1.0}
    params = d.params

    assert set(params.keys()) == {
        "theta",
    }, "The instance of Geometric distribution should have "
    f"parameters {'theta'}, but {params.keys()}. "

    assert (
        params["theta"] == 1.0
    ), f"The parameter theta should be 1.0, but {params['theta']}."


def test_geometric_pmf(data):
    theta = data["geom"]["theta"]
    _y = data["geom"]["rv"]
    pmf_true = data["geom"]["pmf"]
    d = Geometric(theta)
    pmf = d.pmf(_y)

    assert (pmf_true == pmf).all(), "pmf is wrong. "
    f"{pmf_true} is expected, but {pmf} is."


def test_loglikelihood(data):
    theta = data["geom"]["theta"]
    _y = data["geom"]["rv"]
    loglik_true = data["geom"]["loglik"]
    loglik, d = loglikelihood(_y, theta)

    assert (
        "Geometric" == d.__class__.__name__
    ), "It should be an instance of Normal, "
    f"but {d.__class__.__name__}."
    assert loglik_true == loglik, "loglik is wrong. "
    f"{loglik_true} is expected, but {loglik} is."


def test_neg_loglikelihood(data):
    theta = data["geom"]["theta"]
    _y = data["geom"]["rv"]
    loglik_true = data["geom"]["loglik"]
    neg_loglik, _ = neg_loglikelihood(_y, theta)
    neg_loglik_true = (-1.0 * loglik_true) / _y.shape[0]

    assert neg_loglik == neg_loglik_true, "neg_loglik is wrong. "
    f" {neg_loglik_true} is expected, but {neg_loglik} is."


def test_rv_samples(data):
    rv_key = data["key"]
    theta = data["geom"]["theta"]
    _y = data["geom"]["rv"]

    mu = 1 / theta
    var = (1 - theta) / theta**2

    samples, _ = rv_samples(theta, rv_key, _y.shape[0])
    # TODO: Too sensitive to sample size
    # assert (
    #     round((samples.mean() - mu) / mu, 1) < 0.1
    # ), "Mean of geometric distribution should be "
    assert np.isclose(
        round(samples.mean(), 1), mu
    ), "Mean of geometric distribution should be "
    f"close to {mu}, but {round(samples.mean(), 1)}."

    # assert (
    #     round((samples.var() - var) / var, 2) < 0.1
    # )
    assert np.isclose(
        round(samples.var(), 0), var
    ), "std of geometric distribution should be "
    f"close to {round(var, 2)}, but {round(samples.var(), 2)}."


def test_survival_functions(data):
    theta = data["geom"]["theta"]

    survives_ = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    survives = jnp.array([])
    for v in [0, 1, 2, 3, 4]:
        survives = jnp.append(survives, survival_functions(v, theta)[0])

    assert np.allclose(
        survives, survives_
    ), "Average of survival functions values are"
    f"{survives_.mean()}, but {survives.mean()}."
