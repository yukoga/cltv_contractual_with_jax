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
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, loss: Callable = None, X: any = None, y: any = None) -> None:
        """Abstract method for training from data.

        Args:
            loss (function, required): Loss function to be minimized.
                The loss function should take the first argument as
                list of parameters, and other keyword arguments.
            X (any, required): Given feature dataset to train the model.
                Defaults to None.
            y (any, required): Given target variables to train the model.
                Defaults to None.
        """
        pass

        @abstractmethod
        def validate(self, X: any = None, y: any = None) -> any:
            """_summary_

            Args:
                X (any, required):  Given feature dataset to train the model.
                Defaults to None.
                y (any, required): Given target variables to train the model.
                Defaults to None.

            Returns:
                any: Evaluation metrics which shows how far the model is
                    from the true distribution.
            """
            pass
