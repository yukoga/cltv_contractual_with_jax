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


def calc_ltv(y, M, d=0, s: iter = None, model=None):
    def dcf(d, t):
        for k in jnp.arange(t + 1):
            yield 1 / ((1 + d) ** k)

    if s is None:

        def s(t):
            theta = model.dist.params["theta"]
            for k in jnp.arange(t + 1):
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
