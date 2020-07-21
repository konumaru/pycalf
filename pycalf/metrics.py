import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.regression import linear_model

import matplotlib.pyplot as plt
plt.style.use('seaborn')


class StandardDiff():

    def __init__(self):
        self.std_diff = None
        super().__init__()

    def fit(self, X: pd.DataFrame, treatment: pd.Series, weight: np.array = None):
        """Description

        Parameters
        ----------
        X : pd.DataFrame

        treatment : pd.Series

        weight : np.array

        Returns
        -------
        None
        """
        covariates = X.columns.tolist()
        is_treat = (treatment == 1)
        # Treat Group.
        treat_df = X[treatment == 1]
        treat_avg = np.average(treat_df, weights=weight[is_treat], axis=0)
        treat_var = np.average(np.square(treat_df - treat_avg),
                               weights=weight[is_treat], axis=0)
        # Control Group.
        control_df = X[treatment == 0]
        control_avg = np.average(control_df, weights=weight[~is_treat], axis=0)
        control_var = np.average(np.square(control_df - control_avg),
                                 weights=weight[~is_treat], axis=0)
        # Estimate d_value.
        sc = np.sqrt((sum(is_treat) * treat_var + sum(~is_treat) * control_var) / X.shape[0])
        d_value = np.abs(treat_avg - control_avg) / sc
        self.std_diff = pd.Series(d_value, index=covariates).sort_values()

    def transform(self):
        """Description

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
        """
        return self.std_diff

    def fit_transform(self, X: pd.DataFrame, treatment: pd.Series, weight: np.array = None):
        """Description

        Parameters
        ----------
        X : pd.DataFrame

        treatment : pd.Series

        weight : np.array

        Returns
        -------
        pd.DataFrame
        """
        self.fit(X, treatment, weight)
        return self.transform()

    def plot_d_values(self, figsize: tuple = (12, 6), threshold: float = 0.2):
        """Description

        Parameters
        ----------
        figsize : tuple

        threshold : float

        Returns
        -------
        """
        plt.figure(figsize=figsize)
        plt.title('Standard Diff')
        plt.bar(self.std_diff.index, self.std_diff.values)
        plt.ylabel('d value')
        plt.xticks(rotation=90)
        plt.plot([0.0, len(self.std_diff.index)],
                 [threshold, threshold], color='tab:red', linestyle='--')
        plt.tight_layout()
        plt.show()


class AttributeEffect():
    def __init__(self):
        self.effect = None
        super().__init__()

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series, weight: np.array = None):
        """Description

        Parameters
        ----------
        X : pd.DataFrame

        treatment : pd.Series

        y : pd.Series

        weight : np.array

        Returns
        -------
        None
        """
        is_treat = (treatment == 1)
        self.treat_result = sm.WLS(y[is_treat], X[is_treat], weights=weight[is_treat]).fit()
        self.control_result = sm.WLS(y[~is_treat], X[~is_treat], weights=weight[~is_treat]).fit()

    def transform(self):
        """Description

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
            result[f'Z{i}_effect'] = model.params.round(1)
            result[f'Z{i}_tvalue'] = model.tvalues.round(2).apply(
                lambda x: str(x) + '**' if abs(x) >= 1.96 else str(x)
            )

        # Estimate Lift Values
        result['Lift'] = (result['Z1_effect'] - result['Z0_effect'])
        result_df = result.sort_values(by='Lift')
        self.effect = result_df
        return result_df

    def plot_lift_values(self, figsize: tuple = (12, 6)):
        """Description

        Parameters
        ----------
        figsize : tuple

        Returns
        -------
        """
        plt.figure(figsize=figsize)
        plt.title('Treatment Lift Values')
        plt.bar(self.effect.index, self.effect['Lift'].values)
        plt.ylabel('Lift Value')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


class VIF():
    def __init__(self):
        self.result = None

    def fit(self, data: pd.DataFrame):
        """Description

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        None
        """
        vif = pd.DataFrame(index=data.columns.tolist(), columns=['VIF'], dtype='float64')

        for feature in data.columns.tolist():
            X = data.drop([feature], axis=1)
            y = data[feature]

            model = linear_model.OLS(endog=y, exog=X)
            r2 = model.fit().rsquared

            vif.loc[feature, 'VIF'] = np.round(1 / (1 - r2), 2)

        self.result = vif

    def transform(self):
        """Description

        Parameters
        ----------

        Returns
        -------
        pd.DataFrame
        """
        return self.result

    def fit_transform(self, data: pd.DataFrame, **kwargs):
        """Description

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        self.fit(data, **kwargs)
        return self.transform(data)
