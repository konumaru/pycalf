import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn')


class StandardDiff():

    def __init__(self):
        self.std_diff = None
        super().__init__()

    def fit(self, X: pd.DataFrame, treatment: pd.Series, weight: np.array = None):
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
        return self.std_diff

    def fit_transform(self, X: pd.DataFrame, treatment: pd.Series, weight: np.array = None):
        self.fit(X, treatment, weight)
        return self.transform()

    def plot_d_values(self, figsize: tuple = (12, 6), thresh: float = 0.2):
        plt.figure(figsize=figsize)
        plt.title('Standard Diff')
        plt.bar(self.std_diff.index, self.std_diff.values)
        plt.ylabel('d value')
        plt.xticks(rotation=90)
        plt.plot([0.0, len(self.std_diff.index)], [thresh, thresh], color='tab:red', linestyle='--')
        plt.tight_layout()
        plt.show()
