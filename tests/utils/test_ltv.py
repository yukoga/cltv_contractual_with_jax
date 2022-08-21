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
from jax_cltv.utils.ltv import calc_ltv
from jax_cltv.models.mle import MLE
from jax_cltv.dists.geom import Geometric
from jax_cltv.dists.geom import neg_loglikelihood as geom_neg_loglikelihood


def test_calc_ltv_from_data():
    N = 3
    T = 5
    M = 100.0
    d = 0.15
    theta = 0.5

    # for one user's LTV.
    t = jnp.arange(T)
    values = [M * (1 / (1 + d) ** t) * (1 - theta) ** t for t in range(T)]
    ltv = jnp.array(
        [np.sum(values[: t + 1]) for t in range(T)],
        dtype="float32",
    )

    s = jnp.array([1.0, 0.5, 0.5**2, 0.5**3, 0.5**4])
    e_cltv = calc_ltv(t, M, d, s)
    e_cltv = jnp.round(e_cltv.to_py(), 2)
    ltv = jnp.round(ltv.to_py(), 2)

    assert (ltv == e_cltv).all(), f"Expected LTV must be {ltv}, but {e_cltv}."


def test_calc_ltv_from_model(data):
    _x = data["geom"]["rv"]

    def model():
        return MLE(dist=Geometric(theta=0.01))

    def loss(w, X, y):
        nl, _ = geom_neg_loglikelihood(X, w[0])
        return nl

    m = model()
    w_init = jnp.array(list(m.params.values()))
    _ = m.fit(loss, w_init, (_x, None))

    T = 5
    M = 100.0
    d = 0.15
    theta = 0.5

    # for one user's LTV.
    t = jnp.arange(T)
    values = [M * (1 / (1 + d) ** t) * (1 - theta) ** t for t in range(T)]
    ltv = jnp.array(
        [np.sum(values[: t + 1]) for t in range(T)],
        dtype="float32",
    )

    e_cltv = calc_ltv(t, M, d, model=m)
    e_cltv = jnp.round(e_cltv.to_py(), 2)
    ltv = jnp.round(ltv.to_py(), 2)

    assert (ltv == e_cltv).all(), f"Expected LTV must be {ltv}, but {e_cltv}."
