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


from typing import Callable
from jax_cltv.dists.base import BaseContinuousDist, BaseDiscreteDist, BaseDist
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
        self.__params = dist.get_params()
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
        if isinstance(self.dist, BaseContinuousDist):
            res = self.__optimize_unbounded(loss, init_params, data, options)
        elif isinstance(self.dist, BaseDiscreteDist):
            res = self.__optimize_bounded(loss, init_params, data, options)
        else:
            raise TypeError(
                "Probability distirbution must be extended from "
                "either BaseContinousDist or BaseDiscreteDist, "
                f"but {self.dist.__class__.__name__}"
            )
        return res

    def validate(self, X: any = None, y: any = None) -> None:
        pass

    def predict(self, X: any = None):
        pass

    def __optimize_bounded(
        self,
        loss: Callable,
        params: iter,
        data: tuple,
        options: dict = {"gtol": 1e-4},
    ) -> OptimizeResults:
        optimizer = JaxOptOptimizer(options=options)
        return optimizer(loss, params, data)

    def __optimize_unbounded(
        self,
        loss: Callable,
        params: iter,
        data: tuple,
        options: dict = {"gtol": 1e-4},
    ) -> OptimizeResults:
        optimizer = DefaultOptimizer(options=options)
        return optimizer(loss, params, data)
