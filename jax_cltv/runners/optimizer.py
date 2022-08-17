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

from typing import Any, Callable
from abc import ABC
from jax.scipy.optimize import minimize
from jaxopt import ScipyBoundedMinimize
from jax.scipy.optimize import OptimizeResults as jaxOptimizerResults
from jaxopt._src.base import OptStep as optOptimizerResults


class OptimizeResults(object):
    def __init__(self, results: Any):
        """OptimizerResults contains a result of optimization tasks
            wrapped jax.scipy.optimize.OptimizeResults
            or jaxopt._src.base_OptStep object.

        Args:
            results (Any): Optimization result object. The instance of
                jax.scipy.optimize.OptimizationResults
                or jaxopt._src.base.OptStep.
        """
        self.params = None
        self.status = None
        self.success = None
        self.func_val = None
        self.niter = None
        self.parse_results(results)

    def parse_results(self, results: Any) -> None:
        """Get and fit optimization results properties into fields.

        Args:
            results (Any): Optimization result object. The instance of
                jax.scipy.optimize.OptimizationResults
                or jaxopt._src.base.OptStep.

        Raises:
            TypeError: Raises when the results are not the instance of
                jax.scipy.optimize.OptimizationResults or
                jaxopt._src.base.OptStep.
        """
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

    def parse_jax_results(self, results: jaxOptimizerResults) -> None:
        """Fit jax optimization results properties into fields.

        Args:
            results (jaxOptimizerResults): Optimization result object.
                The instance of jax.scipy.optimize.OptimizationResults.
        """
        self.params = results.x
        self.status = results.status
        self.success = results.success
        self.func_val = results.fun
        self.niter = results.nit

    def parse_jaxopt_results(self, results) -> None:
        """Fit jaxopt optimization results properties into fields.

        Args:
            results (optOptimizerResults): Optimization result object.
                The instance of jaxopt._src.base.OptStep.
        """
        self.params = results.params
        self.status = results.state.status
        self.success = results.state.success
        self.func_val = results.state.fun_val
        self.niter = results.state.iter_num


class BaseOptimizer(ABC):
    def __init__(self, options: dict = None):
        self._optimizer = None
        self.options = options

    def __call__(
        self, func: Callable, params: iter, data: tuple
    ) -> OptimizeResults:
        """Make it callable the Optimizer object. Call self.run inside.

        Args:
            func (Callable): Loss function to be optimized.
            params (iter): The list of parameters
                which characterize loss function.
            data (tuple): The data that the optimizer learns from.

        Returns:
            OptimizeResults: Returns optimize results.
        """
        return self.run(func, params, data, self.options)

    @property
    def runner(self) -> Any:
        """Runner property which returns JAX or JAXopt optimizer.

        Returns:
            Any: Optimizer object which is defined in JAX or JAXopt.
        """
        return self._optimizer

    @runner.setter
    def runner(self, runner: Any) -> None:
        """Setter for runner = optimizer.

        Args:
            runner (Any): JAX or JAXopt optimizer.
        """
        self._optimizer = runner

    def run(
        self, func: Callable, params: iter, data: tuple, options: dict = {}
    ) -> OptimizeResults:
        """Execute optimization for given params based on data
            and optimize options.

        Args:
            func (Callable): Loss function to be optimized.
            params (iter): The list of parameters
                which characterize loss function.
            data (tuple): The data that the optimizer learns from.
            options (dict, optional): Optional parameters for optimizations.

        Returns:
            OptimizeResults: Returns optimize results.
        """
        res = self.runner(func, params, data, options)
        return OptimizeResults(res)


class DefaultOptimizer(BaseOptimizer):
    def __init__(self, options: dict = {}):
        super().__init__(options=options)
        self.runner = lambda l, p, d, o: minimize(
            l, p, d, method="BFGS", options=dict(options, **o)
        )


class JaxOptOptimizer(BaseOptimizer):
    def __init__(self, options: dict = {}):
        super().__init__(options=options)

        def minimizer(loss, param, data, opt):
            m = ScipyBoundedMinimize(
                fun=loss,
                method="l-bfgs-b",
                options=dict(options, **opt),
            )
            lb, ub = 0.00001, 0.99999
            return m.run(param, (lb, ub), *data)

        self.runner = minimizer
