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


from typing import Iterable
import pandas as pd
import jax.numpy as jnp
import jax.random as random
from jax_cltv.datasets.bases import BaseDataset
from jax_cltv.dists.geom import rv_samples


class DummySubscriptions(BaseDataset):
    def __init__(self, rng_key=1, p=0.5, size=100, noise=None):
        self.data = self.get_samples(rng_key, p, size, noise)

    def to_pandas(self, columns: Iterable = None) -> pd.DataFrame:
        df = pd.DataFrame(self.data, columns=columns)
        if not columns:
            df.columns = df.columns.map(lambda v: f"Day{v}")
        return df

    def get_samples(
        self,
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
            # rtns += (
            #     round(
            #         noise["mu"]
            #         + noise["sigma"] * random.normal(rng_key, (size,))
            #     )
            #     .to_py()
            #     .astype("int")
            # )
        drtns = (rtns.max() + 1).to_py().astype("int")
        # D = int(drtns)
        for i, r in enumerate(rtns):
            r = int(r)
            if i == 0:
                x = jnp.concatenate(
                    [jnp.full(r - 1, 1), jnp.full(drtns - r + 1, 0)]
                )
            else:
                x = jnp.concatenate(
                    [
                        x,
                        jnp.concatenate(
                            [jnp.full(r, 1), jnp.full(drtns - r, 0)]
                        ),
                    ]
                )
        return x.reshape(size, drtns)

    def load(self):
        """
        Abstract method for loading data from somewhere.
        """
        pass

    def to_csv(
        self, path="./data/cltv_synthetic.csv", columns: Iterable = None
    ) -> None:
        """
        Abstract method to save data as csv file.
        """
        self.to_pandas(columns=columns).to_csv(path, index=False)
