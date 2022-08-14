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


from jax_cltv.models.mle import MLE
from jax_cltv.dists.base import BaseDist
from jax_cltv.dists.normal import Normal


def test_instantiate(data):
    nd = Normal(loc=0.0, scale=1.0)
    m = MLE(dist=nd, params={"loc": 0.0, "scale": 1.0})

    assert "MLE" == m.__class__.__name__, "It should be an instance of MLE, "
    f"but {m.__class__.__name__}."

    assert isinstance(m.dist, BaseDist), "The model has a instance of "
    f"{m.dist.__class__.__name__}, but didn't."

    assert (
        m.dist.__class__.__name__ == "Normal"
    ), "The model has a member of normal distribution, "
    f"but {m.__class__.__name__}."


def test_get_params(data):
    m = MLE(dist="normal", params={"loc": 0.0, "scale": 1.0})
    params = m.params

    assert isinstance(
        params, dict
    ), "Parameters should be an instance of dict, "
    f"but {params.__class__.__name__}"

    assert (
        "loc" in params.keys()
    ), "The member of prob. distribution should have "
    f"parameter loc of {m.dist.__class__.__name__}, but didn't."

    assert (
        "scale" in params.keys()
    ), "The member of prob. distribution should have "
    f"parameter scale of {m.dist.__class__.__name__}, but didn't."


def test_fit(data):
    pass

    # assert (pdf_true == pdf).all(), "pdf is wrong. "
    # f"{pdf_true} is expected, but {pdf} is."


def test_validate(data):
    pass


def test_predict(data):
    pass
