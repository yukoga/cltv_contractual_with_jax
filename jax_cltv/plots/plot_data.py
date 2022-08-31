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
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from jax_cltv.dists.geom import Geometric
from jax_cltv.utils.ltv import calc_ltv, get_survives_from_churns


def plot_chart(
    ax: any,
    data: any = None,
    x: any = None,
    y: any = None,
    yscale: str = None,
    kind: str = "line",
    color: str = None,
    alpha: float = 0.4,
    label: str = None,
    bar_label: str = None,
    **kwargs
):
    if (x is None) and (y is None) and (data is None):
        raise TypeError("Please specify either data or (x and y) parameters.")
    elif type(x) == str and type(y) == str and type(data) == pd.DataFrame:
        x, y = data[x], data[y]
    else:
        pass

    D = x.shape[0]
    N = y.shape[0]

    if kind == "plot":
        ax.plot(x, y, alpha=alpha, label=label, c=color)
    elif kind == "scatter":
        ax.scatter(x, y, alpha=alpha, label=label, c=color)
    elif kind == "hist":
        ax.hist(
            y,
            alpha=alpha,
            label=label,
            color=color,
            bins=kwargs["bins"],
            density=kwargs["density"],
        )
    elif kind == "bar":
        if N != D:
            y = y.value_counts()
        if bar_label:
            bar = ax.bar(x, y, alpha=alpha, label=label, color=color)
            ax.bar_label(bar, label_type=bar_label)
        else:
            ax.bar(x, y, alpha=alpha, label=label, color=color)
    else:
        ax.plot(x, y, alpha=alpha, label=label, c=color)

    if yscale in ("linear", "log", "symlog", "log"):
        ax.set_yscale(yscale)

    ax.set_xticks(x)
    return ax


def plot_churns(
    data: Any,
    theta: Any = None,
    bins: int = None,
    yscale: str = "linear",
    density: bool = False,
    label: str = "Churns",
    style: str = "ggplot",
    figsize: tuple = (16, 9),
    alpha: float = 0.4,
    fontsize: int = 14,
    title: str = "Plot of churned users",
    ax: Any = None,
    bar_label: str = "edge",
) -> plt.Axes:
    """
    Plot churned users tabular data which is in the form as follows:
        - The shape is (# of users, # of days users survies).
        - If an specific user (=i.e. record) churns at a specific day
        (=i.e. column), then the cell must be 1, otherwise 0.

    Parameters
    ----------
    x: array_like
        observed data in the form of 1-D vector.
    theta: array_like
        Shape parameter of a geometric distribution.
        If theta is specified, you can plot pmf of a geometric distribution
        with this parameter.
    bins: int
        Specify bins of churned users histogram. Default: 10.
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
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    N = data.shape[0]
    data = data["churn_dates"].value_counts()
    y = jnp.append(jnp.array([0]), data.values)
    x = jnp.append(jnp.array([0]), jnp.array(data.index))

    if density:
        y = y / N

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Durations")
    ax.set_ylabel("# of churned users.")
    # bar = ax.bar(x, y, yscale=yscale, alpha=alpha, label=label)
    ax = plot_chart(
        ax,
        x=x,
        y=y,
        yscale=yscale,
        kind="bar",
        alpha=alpha,
        density=density,
        label=label,
        bar_label=bar_label,
    )
    if theta:
        gd = Geometric(theta)
        pmf = gd.pmf(x)
        if not density:
            pmf = pmf * N
        ax = plot_chart(
            ax,
            x=x,
            y=pmf,
            yscale=yscale,
            kind="scatter",
            alpha=0.9,
            color="k",
            label=label + " (pmf)",
        )

    plt.legend()
    return ax


def plot_survives(
    data: Any,
    theta: Any = None,
    yscale: str = "linear",
    label: str = "Survives",
    density: bool = False,
    style: str = "ggplot",
    figsize: tuple = (16, 9),
    alpha: float = 0.4,
    fontsize: int = 14,
    title: str = "Plot of survived users",
    ax: Any = None,
    bar_label: str = "edge",
) -> plt.Axes:
    """
    Plot survived users tabular data which is in the form as follows:
        - The shape is (# of users, # of days users survies).
        - If an specific user (=i.e. record) churns at a specific day
        (=i.e. column), then the cell must be 1, otherwise 0.

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

    def subtract_pmf(prev, k):
        if k == 0:
            return prev - pmf.loc[k, "pmf"]
        else:
            return subtract_pmf(prev - pmf.loc[k, "pmf"], k - 1)

    plt.style.use(style)
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    N = data.shape[0]
    y = get_survives_from_churns(data["churn_dates"])
    x = jnp.array(range(y.shape[0]))
    # data = data["churn_dates"].value_counts()
    # day_max = data.index.max() + 1
    # x = jnp.array(range(day_max))
    # y = N - jnp.cumsum(
    #     jnp.array([data[i] if i in data.index else 0 for i in range(day_max)])
    # )

    if density:
        y = y / N

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Durations")
    ax.set_ylabel("# of survived users.")
    ax = plot_chart(
        ax=ax,
        x=x,
        y=y,
        yscale=yscale,
        kind="bar",
        alpha=alpha,
        label=label,
        bar_label=bar_label,
    )
    if theta:
        gd = Geometric(theta)
        pmf = gd.pmf(x)
        pmf = pd.DataFrame(pmf, columns=["pmf"])
        pmf["survived"] = pmf.apply(lambda r: subtract_pmf(1, r.name), axis=1)
        pmf = pd.concat(
            [pd.DataFrame({"pmf": 0, "survived": 1}, index=[0]), pmf],
            axis=0,
            ignore_index=True,
        )
        # pmf = pmf[:-1]
        # sf = gd.sf(x)
        if not density:
            pmf = pmf["survived"] * N
        else:
            pmf = pmf["survived"]
        # ax.scatter(x, pmf, c='k')
        ax = plot_chart(
            ax,
            x=x,
            y=pmf,
            # y=sf,
            yscale=yscale,
            kind="scatter",
            alpha=0.9,
            color="k",
            label=label + " (pmf)",
        )

    plt.legend()
    return ax
