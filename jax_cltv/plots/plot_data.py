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
import matplotlib.pyplot as plt
from jax_cltv.dists.geom import Geometric


def plot_survives(data: any,
                  theta: any=None,
                  bins: int=10,
                  density: bool=False,
                  style: str='ggplot',
                  figsize: tuple=(16, 9),
                  alpha: float=.4,
                  fontsize: int=14,
                  title: str='Plot of survived users') -> plt.Axes:
    """
    Plot survived users tabular data which is in the form as follows:
        - The shape is (# of users, # of days users survies).  
        - If an specific user (=i.e. record) survives at a specific day (=i.e. column),
          then the cell must be 1, otherwise 0.

    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    theta: array_like
        Shape parameter of a geometric distribution.
        If theta is specified, you can plot pmf of a geometric distribution
        with this parameter.
    bins: int
        Specify bins of suvived users histogram. Default: 10.
    density: bool
        Specify if you'd like to plot in relative frequency.
        False = actual number / True = plot relative freq. Default: False
    style: str
        matplotlib style. The default value is ggplot.
    figsize: tuple
        matplotlib figure size. Default: (16, 9)
    alpha: float
        matplotlib degree of transparency 0. - 1. Default: 0.4.
    fontsize: int
        Axes title fontsize. Default: 14.
    title: str
        plot title (specifically, axes title).

    Returns
    -------
    matplotlib.pyplot.Axes
        Log of the probability mass function at data x: jnp.DeviceArray
        and
        the instance of Geometric distribution for given parameters.
    """
    plt.style.use(style)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    N, D = data.shape
    bins = bins

    x = np.linspace(1, D, D)
    data = data.sum(axis=1).astype('int32')
    gd = Geometric(theta)
    pmf = gd.pmf(x)
    if not density:
        pmf = pmf * N

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Durations')
    ax.set_ylabel('# of survived users.')
    ax.hist(data, bins=bins, alpha=alpha, density=density)
    if theta:
        ax.scatter(x, pmf, c='k')
    
    return ax