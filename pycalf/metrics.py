import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm
from statsmodels.regression import linear_model

import matplotlib.pyplot as plt
plt.style.use('seaborn')


class EffectSize():

    def __init__(self):
        self.effect_size = None
        self.effect_name = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, weight: np.ndarray = None):
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
        if weight is None:
            weight = np.ones(X.shape[0])
        # Cauculation Average and Variance of Treat Group.
        treat_avg = np.average(X[treatment], weights=weight[treatment], axis=0)
        treat_var = np.average(np.square(X[treatment] - treat_avg),
                               weights=weight[treatment], axis=0)
        # Cauculation Average and Variance of Control Group.
        control_avg = np.average(X[~treatment], weights=weight[~treatment], axis=0)
        control_var = np.average(np.square(X[~treatment] - control_avg),
                                 weights=weight[~treatment], axis=0)
        # Estimate d_value.
        data_size = X.shape[0]
        treat_size = np.sum(treatment)
        control_size = np.sum(~treatment)
        sc = np.sqrt((treat_size * treat_var + control_size * control_var) / data_size)
        d_value = np.abs(treat_avg - control_avg) / sc

        self.effect_size = np.array(d_value)
        self.effect_name = X.columns.to_numpy()

    def transform(self):
        """Description

        Parameters
        ----------
        None

        Returns
        -------
        tuple
        """
        return (self.effect_name, self.effect_size)

    def fit_transform(self, X: pd.DataFrame, treatment: np.ndarray, weight: np.ndarray = None):
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


def plot_effect_size(
        X, treatment, weight=None,
        ascending=False, sortbyraw=True, figsize=(12, 6), threshold=0.2):
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
    plt.title('Standard Diff')

    plt.bar(raw_names[sorted_index], raw_effects[sorted_index],
            color='tab:blue', label='Raw')
    plt.bar(ajusted_names[sorted_index], ajusted_effects[sorted_index],
            color='tab:cyan', label='Ajusted', width=0.5)
    plt.ylabel('d value')
    plt.xticks(rotation=90)
    plt.plot([0.0, len(raw_names)], [threshold, threshold], color='tab:red', linestyle='--')
    plt.tight_layout()
    plt.legend()
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
        return self.transform()


def plot_roc_curve(y_true, y_score, figsize=(7, 6)):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_probability_distribution(y_true, y_score, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.title('Probability Distoribution.')
    plt.xlabel('Probability')
    plt.ylabel('Number of Data')
    plt.hist(
        y_score[y_true == 0],
        bins=np.linspace(0, 1, 100, endpoint=False),
        rwidth=0.4,
        align='left',
        color='tab:blue'
    )
    plt.hist(
        y_score[y_true == 1],
        bins=np.linspace(0, 1, 100, endpoint=False),
        rwidth=0.4,
        align='mid',
        color='tab:orange'
    )
    plt.show()


def plot_treatment_effect(outcome_name, control_effect, treat_effect, effect_size,
                          figsize=None, fontsize=12):
    plt.figure(figsize=figsize)
    plt.title(outcome_name)
    plt.bar(
        ['control', 'treatment'],
        [control_effect, treat_effect],
        label=f'Treatment Effect : {effect_size}'
    )
    plt.ylabel('effect size')
    plt.legend(loc="upper left", fontsize=fontsize)
    plt.show()


def f1_score(y_true, y_score, threshold='auto'):
    msg = 'mode must be "auto" or 0 to 1.'
    assert threshold == 'auto' or 0 <= threshold < 1, msg

    if threshold == 'auto':
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        gmeans = np.sqrt(tpr * (1 - fpr))
        threshold = thresholds[np.argmax(gmeans)]

    score = metrics.f1_score(y_true, (y_score > threshold))
    return score
