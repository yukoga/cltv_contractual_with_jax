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


# import jax
import jax.numpy as jnp
from jax_cltv.models.mle import MLE
from jax_cltv.dists.base import BaseDist
from jax_cltv.dists.normal import Normal, neg_loglikelihood
from jax_cltv.dists.geom import Geometric
from jax_cltv.dists.geom import neg_loglikelihood as geom_neg_loglikelihood


def test_instantiate(data):
    nd = Normal(loc=0.0, scale=1.0)
    m = MLE(dist=nd)

    assert "MLE" == m.__class__.__name__, "It should be an instance of MLE, "
    f"but {m.__class__.__name__}."

    assert isinstance(m.dist, BaseDist), "The model has a instance of "
    f"{m.dist.__class__.__name__}, but didn't."

    assert (
        m.dist.__class__.__name__ == "Normal"
    ), "The model has a member of normal distribution, "
    f"but {m.__class__.__name__}."


def test_get_params(data):
    nd = Normal(loc=0.0, scale=1.0)
    m = MLE(dist=nd)
    params = m.params

    assert isinstance(
        params, dict
    ), "Parameters should be an instance of dict, "
    f"but {params.__class__.__name__}"

    assert (
        "loc" in params.keys()
    ), "The member of prob. distribution should have "
    f"parameter loc of {m.dist.__class__.__name__}, but didn't."

    assert (
        "scale" in params.keys()
    ), "The member of prob. distribution should have "
    f"parameter scale of {m.dist.__class__.__name__}, but didn't."


def test_fit_unbounded(data):
    _mu = data["normal"]["mu"]
    _sigma = data["normal"]["sigma"]
    _x, _y = data["normal"]["rv"]

    def model():
        return MLE(dist=Normal(loc=0.0, scale=1.0))

    def loss(w, X, y):
        nl, _ = neg_loglikelihood(y, w[0], w[1])
        return nl

    m = model()
    w_init = jnp.array(list(m.params.values()))
    res = m.fit(loss, w_init, (_x, _y))

    assert (
        res.__class__.__name__ == "OptimizeResults"
    ), "The results of fit should be OptimizeResults, "
    f"but {res.__class__.__name__}"

    mu, sigma = res.params[0], res.params[1]
    mu, sigma = round(mu, 2), round(sigma, 2)
    _mu, _sigma = round(_mu, 2), round(_sigma, 2)

    assert res.success, "Optimization has not been succeeded."

    assert (
        round(abs(mu - _mu) / _mu, 2) < 0.1
    ), f"Inferenced parameter mu {mu} doesn't mach the true value {_mu}."
    assert (
        round(abs(sigma - _sigma) / _sigma, 2) < 0.1
    ), f"Inferenced parameter mu {sigma} doesn't mach the true value {_sigma}."


def test_fit_bounded(data):
    _theta = data["geom"]["theta"]
    _x = data["geom"]["rv"]

    def model():
        return MLE(dist=Geometric(theta=0.01))

    def loss(w, X, y):
        nl, _ = geom_neg_loglikelihood(X, w[0])
        return nl

    m = model()
    w_init = jnp.array(list(m.params.values()))
    res = m.fit(loss, w_init, (_x, None))

    assert (
        res.__class__.__name__ == "OptimizeResults"
    ), "The results of fit should be OptimizeResults, "
    f"but {res.__class__.__name__}"

    theta = round(res.params[0], 2)
    _theta = round(_theta, 2)

    assert res.success, "Optimization has not been succeeded."

    assert (
        round(abs(theta - _theta) / _theta, 2) < 0.1
    ), f"Inferenced parameter mu {theta} doesn't mach the true value {_theta}."


def test_validate(data):
    pass


def test_predict(data):
    pass
