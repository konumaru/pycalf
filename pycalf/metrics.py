import numpy as np
import pandas as pd


def StandardDiff(X, treatment, weight=None):
    covariates = X.columns().tolist()
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
    result = pd.Series(d_value, index=covariates).sort_values()
    return result
