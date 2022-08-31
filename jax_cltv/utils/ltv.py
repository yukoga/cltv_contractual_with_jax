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

from typing import Any
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as random
from jax_cltv.dists.geom import rv_samples


def calc_ltv(y, M, d=0, s: iter = None, model=None):
    def dcf(d, t):
        for k in jnp.arange(t + 1):
            yield 1 / ((1 + d) ** k)

    if s is None:

        def s(t):
            theta = model.dist.params["theta"]
            for k in jnp.arange(t + 1):
                # TODO: calc survival function with model.predict.
                yield (1 - theta) ** k

    else:
        _s = s

        def s(t):
            for k in jnp.arange(t + 1):
                yield _s[k]

    ltv = jnp.array([])
    for v in y:
        ltv = jnp.append(
            ltv,
            jnp.dot(
                jnp.array(np.fromiter(dcf(d, v), "float32")),
                jnp.array(np.fromiter(s(v), "float32")),
            )
            * M,
        )

    return ltv


def generate_geom_survive_flags(
    arr_unsuccess: jnp.DeviceArray,
) -> jnp.DeviceArray:
    size = arr_unsuccess.shape[0]
    if isinstance(arr_unsuccess, jnp.DeviceArray):
        n_success = (arr_unsuccess.max()).to_py().astype("int")
    else:
        n_success = (arr_unsuccess.max()).astype("int")
    # n_success = (arr_unsuccess.max() + 1).to_py().astype("int")
    for i, r in enumerate(arr_unsuccess):
        r = int(r)
        if i == 0:
            x = jnp.concatenate(
                [jnp.full(r - 1, 1), jnp.full(n_success - r + 1, 0)]
            )
        else:
            x = jnp.concatenate(
                [
                    x,
                    jnp.concatenate(
                        [jnp.full(r - 1, 1), jnp.full(n_success - r + 1, 0)]
                    ),
                ]
            )
    return x.reshape(size, n_success)


def generate_geom_samples(
    rng_key: jnp.DeviceArray,
    p: jnp.DeviceArray,
    size: int,
    noise: dict = None,
) -> jnp.DeviceArray:
    # arrays each elements indicates the day users churn.
    if not isinstance(rng_key, jnp.DeviceArray):
        rng_key = random.PRNGKey(rng_key)
    rtns, _ = rv_samples(p, rng_key, size=size)
    if noise and len(set(noise.keys()).intersection(set(["lam"]))) == 1:
        rtns += random.poisson(rng_key, noise["lam"], (size,))
    return generate_geom_survive_flags(rtns), rtns
    # drtns = (rtns.max() + 1).to_py().astype("int")
    # for i, r in enumerate(rtns):
    #     r = int(r)
    #     if i == 0:
    #         x = jnp.concatenate(
    #             [jnp.full(r - 1, 1), jnp.full(drtns - r + 1, 0)]
    #         )
    #     else:
    #         x = jnp.concatenate(
    #             [
    #                 x,
    #                 jnp.concatenate([jnp.full(r, 1), jnp.full(drtns - r, 0)]),
    #             ]
    #         )
    # return x.reshape(size, drtns)


def get_survives_from_churns(
    data: jnp.DeviceArray,
) -> jnp.DeviceArray:
    N = data.shape[0]
    if not isinstance(data, pd.Series):
        data = pd.Series(data).value_counts()
    else:
        data = data.value_counts()
    # day_max = (data.index.max() + 1).astype("int32")
    day_max = data.index.max() + 1
    x = jnp.array(range(day_max))
    y = N - jnp.cumsum(
        jnp.array([data[i] if i in data.index else 0 for i in range(day_max)])
    )
    return y
    # return jnp.append(jnp.array([N]), N - jnp.cumsum(counts))
    # x = generate_geom_survive_flags(arr_unsuccess)
    # return x.sum(axis=0).astype("int32")


def get_churns_from_data(arr_unsuccess: jnp.DeviceArray) -> jnp.DeviceArray:
    s = pd.Series(arr_unsuccess).value_counts()
    y = s.values
    x = s.index.values
    return x, y


# def get_geom_final_successes(theta: float, size: int):
#     return rv_samples(theta, size=size)


# def get_geom_unsuccesses(unsccuesses: jnp.DeviceArray):
#     pass
