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
import pandas as pd
import matplotlib.pyplot as plt
from jax_cltv.dists.geom import Geometric


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
        ax.bar(x, y, alpha=alpha, label=label, color=color)
    else:
        ax.plot(x, y, alpha=alpha, label=label, c=color)

    if yscale in ("linear", "log", "symlog", "log"):
        ax.set_yscale(yscale)

    return ax


def plot_churns(
    data: Any,
    theta: Any = None,
    bins: int = None,
    yscale: str = "linear",
    density: bool = False,
    kind: str = "plot",
    style: str = "ggplot",
    figsize: tuple = (16, 9),
    alpha: float = 0.4,
    fontsize: int = 14,
    title: str = "Plot of churned users",
    ax: Any = None,
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
    N, D = data.shape
    bins = D if not bins else bins

    x = np.linspace(1, D, D)
    y = data.sum(axis=1).astype("int32")

    if kind == "bar":
        _df = pd.DataFrame({"day": x}).astype("int32")
        data = pd.concat([_df, y.value_counts()], axis=1)
        data.columns = ["day", "churns"]
        data.loc[0, "churns"] = N
        data.fillna(0, inplace=True)
        x, y = data["day"].values, data["churns"].values
        if density:
            y = y / N
        del _df

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Durations")
    ax.set_ylabel("# of churned users.")
    ax.set_xticks(x)
    ax = plot_chart(
        ax,
        x=x,
        y=y,
        yscale=yscale,
        # kind="hist",
        kind=kind,
        alpha=alpha,
        density=density,
        bins=bins,
        label="Observed churns",
    )
    # ax.hist(data, bins=bins, alpha=alpha, density=density)
    if theta:
        gd = Geometric(theta)
        pmf = gd.pmf(x)
        if not density:
            pmf = pmf * N
        # ax.scatter(x, pmf, c='k')
        ax = plot_chart(
            ax,
            x=x,
            y=pmf,
            yscale=yscale,
            kind="scatter",
            alpha=0.9,
            color="k",
            label="probabilty distributions (pmf)",
        )

    plt.legend()
    return ax


def plot_survives(
    data: Any,
    theta: Any = None,
    bins: int = None,
    yscale: str = "linear",
    density: bool = False,
    style: str = "ggplot",
    figsize: tuple = (16, 9),
    alpha: float = 0.4,
    fontsize: int = 14,
    title: str = "Plot of survived users",
    ax: Any = None,
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

    N, D = data.shape
    bins = D if not bins else bins

    x = np.linspace(1, D, D)
    data = data.sum(axis=0).astype("int32")
    x_plot = np.insert(x[:-1], 0, 0)
    if density:
        data = data / N

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Durations")
    ax.set_ylabel("# of survived users.")
    # ax.bar(x, data, alpha=alpha)
    ax.set_xticks(x_plot)
    ax = plot_chart(
        ax=ax,
        x=x_plot,
        y=data,
        yscale=yscale,
        kind="bar",
        alpha=alpha,
        label="Observed survives",
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
        pmf = pmf[:-1]
        if not density:
            pmf = pmf["survived"] * N
        else:
            pmf = pmf["survived"]
        # ax.scatter(x, pmf, c='k')
        ax = plot_chart(
            ax,
            x=x_plot,
            y=pmf,
            yscale=yscale,
            kind="scatter",
            alpha=0.9,
            color="k",
            label="probabilty distributions (pmf)",
        )

    plt.legend()
    return ax
