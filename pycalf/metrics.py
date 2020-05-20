import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn')


def StandardDiff(X, treatment, weight=None):
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
    d_value = np.abs(treat_avg - control_avg) / np.sqrt((treat_var + control_var) * 0.5)
    std_diff = pd.Series(d_value, index=covariates).sort_values()
    return std_diff


def plot_standard_diff(std_diff, figsize=(12, 6), thresh=0.2):
    plt.figure(figsize=figsize)
    plt.title('Standard Diff')
    plt.bar(std_diff.index, std_diff.values)
    plt.ylabel('d value')
    plt.xticks(rotation=90)
    plt.plot([0.0, len(std_diff.index)], [thresh, thresh], color='tab:red', linestyle='--')
    plt.tight_layout()
    plt.show()
