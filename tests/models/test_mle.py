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
from jax_cltv.models.mle import MLE
from jax_cltv.dists.base import BaseDist
from jax_cltv.dists.normal import Normal
from jax_cltv.dists.normal import neg_loglikelihood as norm_neg_loglikelihood
from jax_cltv.dists.beta import Beta
from jax_cltv.dists.beta import neg_loglikelihood as beta_neg_loglikelihood
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
        nl, _ = norm_neg_loglikelihood(y, w[0], w[1])
        return nl

    m = model()
    w_init = jnp.array(list(m.params.values()))
    res = m.fit(loss, w_init, (_x, _y))

    assert (
        res.__class__.__name__ == "OptimizeResults"
    ), "The results of fit should be OptimizeResults, "
    f"but {res.__class__.__name__}"

    mu, sigma = res.params[0], res.params[1]
    mu_m, sigma_m = m.params["loc"], m.params["scale"]
    mu_d, sigma_d = m.dist.params["loc"], m.dist.params["scale"]
    mu, sigma = round(mu, 2), round(sigma, 2)
    mu_m, sigma_m = round(mu_m, 2), round(sigma_m, 2)
    mu_d, sigma_d = round(mu_d, 2), round(sigma_d, 2)
    _mu, _sigma = round(_mu, 2), round(_sigma, 2)

    assert res.success, "Optimization has not been succeeded."

    assert (
        round(abs(mu - _mu) / _mu, 2) < 0.1
    ), f"Inferenced parameter mu {mu} doesn't mach the true value {_mu}."
    assert (
        round(abs(sigma - _sigma) / _sigma, 2) < 0.1
    ), f"Inferenced parameter mu {sigma} doesn't mach the true value {_sigma}."
    assert (
        round(abs(mu_m - _mu) / _mu, 2) < 0.1
    ), f"Updated model parameter mu {mu_m} doesn't mach the true value {_mu}."
    assert (
        round(abs(sigma_m - _sigma) / _sigma, 2) < 0.1
    ), f"Updated model parameter sigma {sigma_m} "
    f"doesn't mach the true value {_sigma}."
    assert (
        round(abs(mu_d - _mu) / _mu, 2) < 0.1
    ), f"Updated dist parameter mu {mu_d} doesn't mach the true value {_mu}."
    assert (
        round(abs(sigma_d - _sigma) / _sigma, 2) < 0.1
    ), f"Updated dist parameter sigma {sigma_d} "
    f"doesn't mach the true value {_sigma}."


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
    theta_m = round(m.params["theta"], 2)
    theta_d = round(m.dist.params["theta"], 2)
    _theta = round(_theta, 2)

    assert res.success, "Optimization has not been succeeded."

    assert (
        round(abs(theta - _theta) / _theta, 2) < 0.1
    ), f"Inferenced parameter mu {theta} doesn't mach the true value {_theta}."
    assert (
        round(abs(theta_m - _theta) / _theta, 2) < 0.1
    ), f"Updated model parameter theta {theta_m} "
    f"doesn't mach the true value {_theta}."
    assert (
        round(abs(theta_d - _theta) / _theta, 2) < 0.1
    ), f"Updated dist parameter theta {theta_d} "
    f"doesn't mach the true value {_theta}."


def test_validate(data):
    pass


def test_predict_bounded(data):
    _x = data["geom"]["rv"]

    def model():
        return MLE(dist=Geometric(theta=0.01))

    def loss(w, X, y):
        nl, _ = geom_neg_loglikelihood(X, w[0])
        return nl

    m = model()
    w_init = jnp.array(list(m.params.values()))
    _ = m.fit(loss, w_init, (_x, None))

    ret = m.predict(2)
    assert (
        round(abs(ret - 0.25) / 0.25, 2) < 0.1
    ), "Predicted value should be close to 0.25,"
    f" but expected value is {round(ret, 3)}."

    ret = m.predict(3)
    assert (
        round(abs(ret - 0.125) / 0.125, 2) < 0.1
    ), "Predicted value should be close to 0.25,"
    f" but expected value is {round(ret, 3)}."

    ret = m.predict(2, lambda x: x**2)
    assert (
        round(abs(ret - 1.0) / 1.0, 2) < 0.1
    ), "Predicted value should be close to 0.25,"
    f" but expected value of x*:2 is {round(ret, 3)}."


def test_predict_unbounded(data):
    _x = data["beta"]["rv"]

    def model():
        return MLE(dist=Beta(a=2.0, b=3.0))

    def loss(w, X, y):
        nl, _ = beta_neg_loglikelihood(X, w[0], w[1])
        return nl

    m = model()
    w_init = jnp.array(list(m.params.values()))
    _ = m.fit(loss, w_init, (_x, None), {"gtol": 1e-4})

    ret = m.predict(0.5)
    assert (
        round(abs(ret - 1.0) / 1.0, 2) < 0.1
    ), "Predicted value should be close to 1.0,"
    f" but expected value is {round(ret, 3)}."

    ret = m.predict(0.3, lambda x: x**2)
    assert (
        round(abs(ret - 0.09) / 0.09, 2) < 0.1
    ), "Predicted value should be close to 0.25,"
    f" but expected value of x*:2 is {round(ret, 3)}."
