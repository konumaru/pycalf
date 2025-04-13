from typing import Dict, Optional, Union

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
        self.effect_size: Optional[np.ndarray] = None
        self.effect_name: Optional[np.ndarray] = None

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
        # Calculation Average and Variance of Treat Group.
        treat_avg = np.average(X[treatment], weights=weight[treatment], axis=0)
        treat_var = np.average(
            np.square(X[treatment] - treat_avg),
            weights=weight[treatment],
            axis=0,
        )
        # Calculation Average and Variance of Control Group.
        control_avg = np.average(X[~treatment], weights=weight[~treatment], axis=0)
        control_var = np.average(
            np.square(X[~treatment] - control_avg),
            weights=weight[~treatment],
            axis=0,
        )
        # Estimate d_value.
        data_size = X.shape[0]
        treat_size = np.sum(treatment)
        control_size = np.sum(~treatment)
        sc = np.sqrt((treat_size * treat_var + control_size * control_var) / data_size)
        d_value = np.abs(treat_avg - control_avg) / sc

        self.effect_size = d_value
        self.effect_name = X.columns.to_numpy()

    def transform(self) -> Dict[str, np.ndarray]:
        """Apply the calculating the effect size d.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 'effect_name' and 'effect_size' arrays.
        """
        if self.effect_name is None or self.effect_size is None:
            raise ValueError("Model not fitted. Call fit() before transform().")
        return {
            "effect_name": self.effect_name,
            "effect_size": self.effect_size,
        }

    def fit_transform(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        weight: Union[np.ndarray, None] = None,
    ) -> Dict[str, np.ndarray]:
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
        self.result: pd.DataFrame
        self.treat_result: Optional[
            sm.regression.linear_model.RegressionResultsWrapper
        ] = None
        self.control_result: Optional[
            sm.regression.linear_model.RegressionResultsWrapper
        ] = None

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

    def transform(self) -> pd.DataFrame:
        """Apply the estimating the effect of the intervention by attribute.

        Returns
        -------
        pd.DataFrame
        """
        if self.treat_result is None or self.control_result is None:
            raise ValueError("Model not fitted. Call fit() before transform().")

        self.result = pd.DataFrame()
        models = [self.control_result, self.treat_result]
        for i, model in enumerate(models):
            self.result[f"Z{i}_effect"] = model.params.round(1)
            self.result[f"Z{i}_tvalue"] = model.tvalues.round(2).apply(
                lambda x: str(x) + "**" if abs(x) >= 1.96 else str(x)
            )

        # Estimate Lift Values
        self.result["Lift"] = self.result["Z1_effect"] - self.result["Z0_effect"]
        self.result.sort_values(by="Lift", inplace=True)
        return self.result


class VIF:
    """Variance Inflation Factor (VIF)."""

    def __init__(self) -> None:
        self.result: pd.DataFrame

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

    def transform(self) -> pd.DataFrame:
        """Apply the calculating vif.

        Returns
        -------
        result : pd.DataFrame
        """
        return self.result

    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
) -> float:
    """Calculate the F1 score.

    Parameters
    ----------
    y_true : numpy.ndarray
        The target vector.
    y_score : numpy.ndarray
        The score vector.
    threshold : float
        Threshold on the decision function used to compute precision and recall.
        Default is 0.5.
    is_auto : bool
        If True, automatically find optimal threshold. Default is True.

    Returns
    -------
    score : float
        F1 score.
    """
    assert 0 <= threshold < 1, "mode must be or 0 to 1."

    if is_auto:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        gmeans = np.sqrt(tpr * (1 - fpr))
        threshold = thresholds[np.argmax(gmeans)]

    score = float(metrics.f1_score(y_true, (y_score > threshold)))
    return score
