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
from jax_cltv.dists.normal import Normal, neg_loglikelihood
from jax_cltv.models.mle import MLE
from jax_cltv.runners.optimizer import (
    DefaultOptimizer,
    JaxOptOptimizer,
    OptimizeResults,
)
from jaxopt import ScipyMinimize


def test_instantiate(data):
    dopt = DefaultOptimizer()
    jopt = JaxOptOptimizer()
    # ores = OptimizeResults()

    assert (
        dopt.__class__.__name__ == "DefaultOptimizer"
    ), "Failed to instantiate DefaultOptimizer."
    assert (
        jopt.__class__.__name__ == "JaxOptOptimizer"
    ), "Failed to instantiate DefaultOptimizer."
    # assert (
    #     ores.__class__.__name__ == "OptimizeResults"
    # ), "Faile to instantiate OptimizeResults"


def test_default_optimizer_resuts(data):
    _mu = data["normal"]["mu"]
    _sigma = data["normal"]["sigma"]
    _x, _y = data["normal"]["rv"]

    def model():
        return MLE(dist=Normal(loc=0.0, scale=1.0))

    def loss(w, X, y):
        nl, _ = neg_loglikelihood(y, w[0], w[1])
        return nl

    m = model()
    res = m.fit(loss, _x, _y)

    ores = OptimizeResults(res)
    mu, sigma = ores.params[0], ores.params[1]

    assert (
        ores.success.dtype == "bool"
    ), "OptimizeResults does not hold correct success flag, "
    f"but {ores.success}"

    assert (
        round(abs(mu - _mu) / mu, 2) < 0.1
    ), f"Inferenced parameter mu {mu} doesn't mach the true value {_mu}."

    assert (
        round(abs(sigma - _sigma) / sigma, 2) < 0.1
    ), f"Inferenced parameter mu {sigma} doesn't mach the true value {_sigma}."

    assert (
        ores.niter.dtype == "int32"
    ), "It seems OptimizeResults.niter does not have the correct value. "
    f"It's type is {ores.niter.dtype}."

    assert (
        ores.status.dtype == "int32"
    ), "It seems OptimizeResults.status does not have the correct value. "
    f"It's type is {ores.status.dtype}."

    assert (
        ores.func_val.dtype == "float32"
    ), "It seems OptimizeResults.func_val does not have the correct value. "
    f"It's type is {ores.func_val.dtype}."


def test_jaxopt_optimizer_resuts(data):
    _mu = data["normal"]["mu"]
    _sigma = data["normal"]["sigma"]
    _x, _y = data["normal"]["rv"]

    def loss(w, data):
        _, y = data
        nl, _ = neg_loglikelihood(y, w[0], w[1])
        return nl

    optimizer = ScipyMinimize(fun=loss, method="bfgs")
    w_init = jnp.append(jnp.zeros((1,)), 1.0)
    res = optimizer.run(w_init, data=(_x, _y))

    ores = OptimizeResults(res)
    mu, sigma = ores.params[0], ores.params[1]

    assert isinstance(
        ores.success, bool
    ), "OptimizeResults does not hold correct success flag, "
    f"but {ores.success}"

    assert (
        round(abs(mu - _mu) / mu, 2) < 0.1
    ), f"Inferenced parameter mu {mu} doesn't mach the true value {_mu}."

    assert (
        round(abs(sigma - _sigma) / sigma, 2) < 0.1
    ), f"Inferenced parameter mu {sigma} doesn't mach the true value {_sigma}."

    assert isinstance(
        ores.niter, int
    ), "It seems OptimizeResults.niter does not have the correct value. "
    f"It's type is {type(ores.niter)}."

    assert isinstance(
        ores.status, int
    ), "It seems OptimizeResults.status does not have the correct value. "
    f"It's type is {type(ores.status)}."

    assert (
        ores.func_val.dtype == "float32"
    ), "It seems OptimizeResults.func_val does not have the correct value. "
    f"It's type is {ores.func_val.dtype}"
