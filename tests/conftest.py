import os
import sys
from typing import Tuple

import pandas as pd
import pytest

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../pycalf/")
)


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    filepath = os.path.join(os.path.dirname(__file__), "data", "q_data_x.csv")
    data = pd.read_csv(filepath)

    covariate_cols = [
        "TVwatch_day",
        "age",
        "sex",
        "marry_dummy",
        "child_dummy",
        "inc",
        "pmoney",
        "area_kanto",
        "area_tokai",
        "area_keihanshin",
        "job_dummy1",
        "job_dummy2",
        "job_dummy3",
        "job_dummy4",
        "job_dummy5",
        "job_dummy6",
        "job_dummy7",
        "fam_str_dummy1",
        "fam_str_dummy2",
        "fam_str_dummy3",
        "fam_str_dummy4",
    ]
    outcome_cols = ["gamecount", "gamedummy", "gamesecond"]
    treatment_col = "cm_dummy"

    X = data[covariate_cols]
    y = data[outcome_cols]
    treatment = data[treatment_col].astype(bool)
    return X, y, treatment
