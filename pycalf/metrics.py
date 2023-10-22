from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from statsmodels.regression import linear_model


class EffectSize:
    """Calculating the effect size-d.

    Examples
    --------
    ate_weight = model.get_weight(treatment, mode='ate')
    es = metrics.EffectSize()
    es.fit(X, treatment, weight=ate_weight)
    es.transform() # return (effect_size, effect_name)
    """

    def __init__(self) -> None:
        self.effect_size = None
        self.effect_name = None

    def fit(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        weight: Union[np.ndarray, None] = None,
    ) -> None:
        """Fit the model with X.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates for propensity score.
        treatment : pd.Series
            Flags with or without intervention.
        weight : np.array
            The weight of each sample.

        Returns
        -------
        None
        """
        if weight is None:
            weight = np.ones(X.shape[0])
        # Cauculation Average and Variance of Treat Group.
        treat_avg = np.average(X[treatment], weights=weight[treatment], axis=0)
        treat_var = np.average(
            np.square(X[treatment] - treat_avg),
            weights=weight[treatment],
            axis=0,
        )
        # Cauculation Average and Variance of Control Group.
        control_avg = np.average(
            X[~treatment], weights=weight[~treatment], axis=0
        )
        control_var = np.average(
            np.square(X[~treatment] - control_avg),
            weights=weight[~treatment],
            axis=0,
        )
        # Estimate d_value.
        data_size = X.shape[0]
        treat_size = np.sum(treatment)
        control_size = np.sum(~treatment)
        sc = np.sqrt(
            (treat_size * treat_var + control_size * control_var) / data_size
        )
        d_value = np.abs(treat_avg - control_avg) / sc

        self.effect_size = d_value
        self.effect_name = X.columns.to_numpy()

    def transform(self):
        """Apply the calculating the effect size d.

        Returns
        -------
        (effect_name, effect_size) : tuple
        """
        return (self.effect_name, self.effect_size)

    def fit_transform(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        weight: Union[np.ndarray, None] = None,
    ):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates for propensity score.
        treatment : pd.Series
            Flags with or without intervention.
        weight : np.array
            The weight of each sample.

        Returns
        -------
        pd.DataFrame
        """
        self.fit(X, treatment, weight)
        return self.transform()


class AttributeEffect:
    """Estimating the effect of the intervention by attribute."""

    def __init__(self) -> None:
        self.effect = None

    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        y: pd.Series,
        weight: Union[np.ndarray, None] = None,
    ) -> None:
        """Fit the model with X, y and weight.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates for propensity score.
        treatment : pd.Series
            Flags with or without intervention.
        y : pd.Series
            Outcome variables.
        weight : np.array
            The weight of each sample.

        Returns
        -------
        None
        """
        if weight is None:
            weight = np.ones(X.shape[0])

        is_treat = treatment == 1

        self.treat_result = sm.WLS(
            y[is_treat],
            X[is_treat],
            weights=weight[is_treat],  # type: ignore
        ).fit()
        self.control_result = sm.WLS(
            y[~is_treat],
            X[~is_treat],
            weights=weight[~is_treat],  # type: ignore
        ).fit()

    def transform(self):
        """Apply the estimating the effect of the intervention by attribute.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
        """
        result = pd.DataFrame()
        models = [self.control_result, self.treat_result]
        for i, model in enumerate(models):
            result[f"Z{i}_effect"] = model.params.round(1)
            result[f"Z{i}_tvalue"] = model.tvalues.round(2).apply(
                lambda x: str(x) + "**" if abs(x) >= 1.96 else str(x)
            )

        # Estimate Lift Values
        result["Lift"] = result["Z1_effect"] - result["Z0_effect"]
        result_df = result.sort_values(by="Lift")
        self.effect = result_df
        return result_df

    def plot_lift_values(self, figsize: Tuple[float, float] = (12, 6)) -> None:
        # TODO: Move to visualize.py
        """Plot the effect.

        Parameters
        ----------
        figsize : tuple
            Figure dimension ``(width, height)`` in inches.

        Returns
        -------
        """
        plt.figure(figsize=figsize)
        plt.title("Treatment Lift Values")
        plt.bar(self.effect.index, self.effect["Lift"].values)  # type: ignore
        plt.ylabel("Lift Value")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


class VIF:
    """Variance Inflation Factor (VIF)."""

    def __init__(self) -> None:
        self.result = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model with data.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        None
        """
        vif = pd.DataFrame(
            index=data.columns.tolist(), columns=["VIF"], dtype="float64"
        )

        for feature in data.columns.tolist():
            X = data.drop([feature], axis=1)
            y = data[feature]

            model = linear_model.OLS(endog=y, exog=X)
            r2 = model.fit().rsquared
            vif.loc[feature, "VIF"] = np.round(1 / (1 - r2), 2)
        self.result = vif

    def transform(self):
        """Apply the calculating vif.

        Returns
        -------
        result : pd.DataFrame
        """
        return self.result

    def fit_transform(self, data: pd.DataFrame, **kwargs):
        """Fit the model with data and apply the calculating vif.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        result : pd.DataFrame
        """
        self.fit(data, **kwargs)
        return self.transform()


def f1_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    is_auto: bool = True,
):
    """Calculate the F1 score.

    Parameters
    ----------
    y_true : numpy.ndarray
        The target vector.
    y_score : numpy.ndarray
        The score vector.
    threshold : 'auto' or float
        Increasing thresholds on the decision function
        used to compute precision and recall.

    Returns
    -------
    score : float
    """
    assert 0 <= threshold < 1, "mode must be or 0 to 1."

    if is_auto:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        gmeans = np.sqrt(tpr * (1 - fpr))
        threshold = thresholds[np.argmax(gmeans)]

    score = metrics.f1_score(y_true, (y_score > threshold))
    return score
