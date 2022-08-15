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
from abc import ABC
from jax.scipy.optimize import OptimizeResults as jaxOptimizerResults
from jaxopt._src.base import OptStep as optOptimizerResults


class OptimizeResults:
    def __init__(self, results: any = None) -> dict:
        self.params = None
        self.status = None
        self.success = None
        self.func_val = None
        self.niter = None
        self.parse_results(results)

    def parse_results(self, results) -> None:
        if isinstance(results, jaxOptimizerResults):
            self.parse_jax_results(results)
        elif isinstance(results, optOptimizerResults):
            self.parse_jaxopt_results(results)
        else:
            raise TypeError(
                "Results object should be instance of either "
                "jax.scipy.optimize.OptimizeResults "
                f"or jaxopt._src.base.Optstep, but {type(results)}."
            )

    def parse_jax_results(self, results):
        self.params = results.x
        self.status = results.status
        self.success = results.success
        self.func_val = results.fun
        self.niter = results.nit

    def parse_jaxopt_results(self, results):
        self.params = results.params
        self.status = results.state.status
        self.success = results.state.success
        self.func_val = results.state.fun_val
        self.niter = results.state.iter_num


class BaseOptimizer(ABC):
    def __init__(
        self,
        loss: Callable = None,
        optimizer: any = None,
    ) -> None:
        self._optimizer = optimizer
        self._loss = loss

    def __call__(self, params: iter, data: tuple, options: dict = None):
        """Make it callable the Optimizer object. Call self.run inside.

        Args:
            params (iter): _description_
            data (tuple): _description_
            options (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self.run(params, data, options)

    @property
    def runner(self) -> any:
        """Runner property which returns JAX or JAXopt optimizer.

        Returns:
            Any: Optimizer object which is defined in JAX or JAXopt.
        """
        return self._optimizer

    @runner.setter
    def set_runner(self, runner: any) -> None:
        """Setter for runner = optimizer.

        Args:
            runner (any): JAX or JAXopt optimizer.
        """
        self.__optimizer = runner

    def run(self, params: iter, data: tuple, options: dict = None) -> any:
        """Execute optimization for given params based on data
            and optimize options.

        Args:
            params (iter): The list of parameters to be optimzied.
            data (tuple): The data that the optimizer learns from.
            options (dict, optional): Optional parameters for optimizations.

        Returns:
            Any: Returns optimize results
        """
        return self.runner(params, data, options)


class DefaultOptimizer(BaseOptimizer):
    pass


class JaxOptOptimizer(BaseOptimizer):
    pass
