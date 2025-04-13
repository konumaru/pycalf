from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn import metrics

from .metrics import EffectSize


def plot_effect_size(
    X: pd.DataFrame,
    treatment: np.ndarray,
    weight: Optional[np.ndarray] = None,
    ascending: bool = False,
    sortbyraw: bool = True,
    figsize: Tuple[float, float] = (12, 6),
    threshold: float = 0.1,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the effects of the intervention.

    Parameters
    ----------
    X : pd.DataFrame
        Covariates for propensity score.
    treatment : numpy.ndarray
        Flags with or without intervention.
    weight : numpy.ndarray, optional
        The weight of each sample. Default is None.
    ascending : bool
        Sort in ascending order.
    sortbyraw : bool
        Flags with sort by raw data or weighted data.
    figsize : tuple
        Figure dimension ``(width, height)`` in inches.
    threshold : float
        Threshold value for effect size.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    es = EffectSize()
    es.fit(X, treatment, weight=weight)
    adjusted_result = es.transform()
    adjusted_names = adjusted_result["effect_name"]
    adjusted_effects = adjusted_result["effect_size"]

    es = EffectSize()
    es.fit(X, treatment, weight=None)
    raw_result = es.transform()
    raw_names = raw_result["effect_name"]
    raw_effects = raw_result["effect_size"]

    sort_data = raw_effects if sortbyraw else adjusted_effects

    if ascending:
        sorted_index = np.argsort(sort_data)
    else:
        sorted_index = np.argsort(sort_data)[::-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title("Standard Diff")

    ax.bar(
        raw_names[sorted_index],
        raw_effects[sorted_index],
        color="tab:blue",
        label="Raw",
    )
    ax.bar(
        adjusted_names[sorted_index],
        adjusted_effects[sorted_index],
        color="tab:cyan",
        label="Adjusted",
        width=0.5,
    )
    ax.set_ylabel("d value")
    ax.set_xticklabels(raw_names[sorted_index], rotation=90)
    ax.plot(
        [0.0, len(raw_names)],
        [threshold, threshold],
        color="tab:red",
        linestyle="--",
    )
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    figsize: Tuple[float, float] = (7, 6),
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot the roc curve.

    Parameters
    ----------
    y_true : numpy.ndarray
        The target vector.
    y_score : numpy.ndarray
        The score vector.
    figsize : tuple
        Figure dimension ``(width, height)`` in inches.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % auc,
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax


def plot_probability_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot propensity scores, color-coded by
    the presence or absence of intervention.

    Parameters
    ----------
    y_true : numpy.ndarray
        The target vector.
    y_score : numpy.ndarray
        The score vector.
    figsize : tuple
        Figure dimension ``(width, height)`` in inches.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    bins_list = list(np.linspace(0, 1, 100, endpoint=False))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title("Probability Distoribution.")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Number of Data")
    ax.hist(
        y_score[y_true == 0],
        bins=bins_list,
        rwidth=0.4,
        align="left",
        color="tab:blue",
    )
    ax.hist(
        y_score[y_true == 1],
        bins=bins_list,
        rwidth=0.4,
        align="mid",
        color="tab:orange",
    )

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax


def plot_treatment_effect(
    outcome_name: str,
    control_effect: Union[float, int],
    treat_effect: Union[float, int],
    effect_size: Union[float, int],
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: int = 12,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot the effects of the intervention.

    Parameters
    ----------
    outcome_name : str
        Outcome name. it use for figure title.
    control_effect : float or int
        Average control Group Effect size.
    treat_effect : float or int
        Average treatment Group Effect size.
    effect_size : float or int
        Treatment Effect size.
    figsize : tuple, optional
        Figure dimension ``(width, height)`` in inches. Default is None.
    fontsize: int
        The font size of the text. See `.Text.set_size` for possible values.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(outcome_name)
    ax.bar(
        ["control", "treatment"],
        [control_effect, treat_effect],
        label=f"Treatment Effect : {effect_size}",
    )
    ax.set_ylabel("effect size")
    ax.legend(loc="upper left", fontsize=fontsize)

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax


def plot_auuc(
    uplift_score: np.ndarray,
    lift: np.ndarray,
    baseline: np.ndarray,
    auuc: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot Area Under the Uplift Curve (AUUC).

    Parameters
    ----------
    uplift_score : numpy.ndarray
        Array of uplift scores.
    lift : numpy.ndarray
        Array of lift, treatment effect.
    baseline : numpy.ndarray
        Array of random treat effect.
    auuc : float, optional
        AUUC score. Default is None.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    label = f"AUUC = {auuc:.4f}" if auuc is not None else None

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title("AUUC")
    ax.plot(lift, label=label)
    ax.plot(baseline)
    ax.set_xlabel("uplift score rank")
    ax.set_ylabel("lift")
    ax.legend(loc="lower right")

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax


def plot_lift_values(
    labels: List[str],
    values: List[Union[float, int]],
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot the lift values.

    Parameters
    ----------
    labels : List[str]
        Labels for x-axis.
    values : List[float or int]
        Values for y-axis.
    figsize : tuple
        Figure dimension ``(width, height)`` in inches. Default is (12, 6).
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    assert len(labels) == len(values), (
        "The length of labels and values must be the same."
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title("Treatment Lift Values")
    ax.bar(labels, values)
    ax.set_ylabel("Lift Value")
    ax.set_xticklabels(labels, rotation=90)

    if ax is None:
        plt.tight_layout()
        plt.show()

    return ax
