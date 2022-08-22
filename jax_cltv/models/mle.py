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

from typing import Callable, Any
from jax_cltv.dists import is_continuous_dist
from jax_cltv.dists.base import BaseDist
from jax_cltv.models.base import BaseModel
from jax_cltv.runners.optimizer import (
    DefaultOptimizer,
    JaxOptOptimizer,
    OptimizeResults,
)


class MLE(BaseModel):
    def __init__(
        self, dist: BaseDist = None, lr_params: dict = None
    ) -> BaseModel:
        self.dist = dist
        self.__params = dist.params
        self.lr_params = lr_params

    @property
    def params(self) -> dict:
        return self.__params

    @params.setter
    def params(self, params: dict = None) -> None:
        self.__params = params

    def fit(
        self,
        loss: Callable,
        init_params: iter,
        data: tuple,
        options: dict = {"gtol": 1e-4},
    ) -> OptimizeResults:
        if is_continuous_dist(self.dist):
            res = self.optimize(
                DefaultOptimizer(options), loss, init_params, data
            )
        else:
            res = self.optimize(
                JaxOptOptimizer(options), loss, init_params, data
            )

        self.params = {
            k: res.params[i] for i, (k, v) in enumerate(self.params.items())
        }
        self.dist = self.dist.__class__(**self.params)
        return res

    def validate(self, X: any = None, y: any = None) -> None:
        pass

    def predict(self, X: any = None, func: Callable = None):
        if not func:

            def func(X):
                return 1

        if is_continuous_dist(self.dist):
            expt = func(X) * self.dist.pdf(X)
        else:
            expt = func(X) * self.dist.pmf(X)

        return expt

    def optimize(
        self,
        optimizer: Any,
        loss: Callable,
        params: iter,
        data: tuple,
    ) -> OptimizeResults:
        return optimizer(loss, params, data)
