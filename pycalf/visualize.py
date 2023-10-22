from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from .metrics import EffectSize


def plot_effect_size(
    X: np.ndarray,
    treatment: np.ndarray,
    weight: np.ndarray | None = None,
    ascending: bool = False,
    sortbyraw: bool = True,
    figsize: Tuple[float, float] = (12, 6),
    threshold: float = 0.2,
) -> None:
    """
    Plot the effects of the intervention.

    Parameters
    ----------
    X : numpy.ndarray
        Covariates for propensity score.
    treatment : numpy.ndarray
        Flags with or without intervention.
    weight : numpy.ndarray
        The weight of each sample
    ascending : bool
        Sort in ascending order.
    sortbyraw : bool
        Flags with sort by raw data or weighted data.
    figsize : tuple
        Figure dimension ``(width, height)`` in inches.
    threshold : float
        Threshold value for effect size.

    Returns
    -------
    None
    """
    es = EffectSize()
    es.fit(X, treatment, weight=weight)
    ajusted_names, ajusted_effects = es.transform()

    es = EffectSize()
    es.fit(X, treatment, weight=None)
    raw_names, raw_effects = es.transform()

    sort_data = raw_effects if sortbyraw else ajusted_effects

    if ascending:
        sorted_index = np.argsort(sort_data)
    else:
        sorted_index = np.argsort(sort_data)[::-1]

    plt.figure(figsize=figsize)
    plt.title("Standard Diff")

    plt.bar(
        raw_names[sorted_index],
        raw_effects[sorted_index],
        color="tab:blue",
        label="Raw",
    )
    plt.bar(
        ajusted_names[sorted_index],
        ajusted_effects[sorted_index],
        color="tab:cyan",
        label="Ajusted",
        width=0.5,
    )
    plt.ylabel("d value")
    plt.xticks(rotation=90)
    plt.plot(
        [0.0, len(raw_names)],
        [threshold, threshold],
        color="tab:red",
        linestyle="--",
    )
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_roc_curve(y_true, y_score, figsize=(7, 6)) -> None:
    """Plot the roc curve.

    Parameters
    ----------
    y_true : numpy.ndarray
        The target vector.
    y_score : numpy.ndarray
        The score vector.
    figsize : tuple
        Figure dimension ``(width, height)`` in inches.

    Returns
    -------
    None
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_probability_distribution(y_true, y_score, figsize=(12, 6)) -> None:
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

    Returns
    -------
    None
    """
    bins_list = list(np.linspace(0, 1, 100, endpoint=False))
    plt.figure(figsize=figsize)
    plt.title("Probability Distoribution.")
    plt.xlabel("Probability")
    plt.ylabel("Number of Data")
    plt.hist(
        y_score[y_true == 0],
        bins=bins_list,
        rwidth=0.4,
        align="left",
        color="tab:blue",
    )
    plt.hist(
        y_score[y_true == 1],
        bins=bins_list,
        rwidth=0.4,
        align="mid",
        color="tab:orange",
    )
    plt.show()


def plot_treatment_effect(
    outcome_name,
    control_effect,
    treat_effect,
    effect_size,
    figsize=None,
    fontsize=12,
) -> None:
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
    figsize : tuple
        Figure dimension ``(width, height)`` in inches.
    fontsize: int
        The font size of the text. See `.Text.set_size` for possible values.

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    plt.title(outcome_name)
    plt.bar(
        ["control", "treatment"],
        [control_effect, treat_effect],
        label=f"Treatment Effect : {effect_size}",
    )
    plt.ylabel("effect size")
    plt.legend(loc="upper left", fontsize=fontsize)
    plt.show()


def plot_auuc(uplift_score, lift, baseline, auuc=None) -> None:
    """Plot Area Under the Uplift Curve (AUUC).

    Parameters
    ----------
    uplift_score : numpy.ndarray
        Array of uplift scores.
    lift : numpy.ndarray
        Array of lift, treatment effect.
    baseline : numpy.ndarray
        Array of random treat effect.
    auuc : float
        AUUC score.

    Returns
    -------
    None
    """
    label = f"AUUC = {auuc:.4f}" if auuc is not None else None

    plt.title("AUUC")
    plt.plot(lift, label=label)
    plt.plot(baseline)
    plt.xlabel("uplift score rank")
    plt.ylabel("lift")
    plt.legend(loc="lower right")
    plt.show()
