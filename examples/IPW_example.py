import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from pycalf import PropensityScore


def main():
    # Load Sample Data.
    # Reference by
    # https://raw.githubusercontent.com/iwanami-datascience/vol3/master/kato%26hoshino/q_data_x.csv
    df = pd.read_csv('sample/q_data_x.csv')

    # Define variables required for inference.
    covariate_cols = [
        'TVwatch_day', 'age', 'sex', 'marry_dummy', 'child_dummy', 'inc', 'pmoney',
        'area_kanto', 'area_tokai', 'area_keihanshin', 'job_dummy1', 'job_dummy2',
        'job_dummy3', 'job_dummy4', 'job_dummy5', 'job_dummy6', 'job_dummy7',
        'fam_str_dummy1', 'fam_str_dummy2', 'fam_str_dummy3', 'fam_str_dummy4'
    ]
    outcome_cols = ['gamecount', 'gamedummy', 'gamesecond']
    treatment_col = 'cm_dummy'

    # Define IPW Class.
    learner = LogisticRegression(** {
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42
    })
    model = PropensityScore.IPW(learner)

    # Set Values from dataframe.
    X = df[covariate_cols]
    y = df[outcome_cols]
    treatment = df[treatment_col]

    # Scaling Raw Data.
    scaler = preprocessing.MinMaxScaler()
    scaled_X = scaler.fit_transform(X)

    # Fit model.
    model.fit(scaled_X, treatment)

    # Inference Some Effect.
    print('ATE:')
    print(model.estimate_ate(treatment, y), '\n')

    print('ATT:')
    print(model.estimate_att(treatment, y), '\n')

    print('ATU:')
    print(model.estimate_atu(treatment, y), '\n')


if __name__ == '__main__':
    main()
