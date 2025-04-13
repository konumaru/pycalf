# pycalf: Causal Inference Library

[![Documentation Status](https://readthedocs.org/projects/pycalf/badge/?version=latest)](https://pycalf.readthedocs.io/en/latest/?badge=latest)
[![PyPI Latest Release](https://img.shields.io/pypi/v/pycalf.svg)](https://pypi.org/project/pycalf/)
[![codecov](https://codecov.io/gh/konumaru/pycalf/graph/badge.svg?token=A424F14MTQ)](https://codecov.io/gh/konumaru/pycalf)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A library to make causal inference easier by using Python

## Getting Started

### Installing

```shell
pip install pycalf
```

## Example

### Inference with IPW

```python
from pycalf import metrics
from pycalf.propensity_score import IPW

# Load Data and Define Variables
df = pd.read_csv('sample/q_data_x.csv')
covariate_cols = [
    'TVwatch_day', 'age', 'sex', 'marry_dummy', 'child_dummy', 'inc', 'pmoney',
    'area_kanto', 'area_tokai', 'area_keihanshin', 'job_dummy1', 'job_dummy2',
    'job_dummy3', 'job_dummy4', 'job_dummy5', 'job_dummy6', 'job_dummy7',
    'fam_str_dummy1', 'fam_str_dummy2', 'fam_str_dummy3', 'fam_str_dummy4'
]
outcome_cols = ['gamecount', 'gamedummy', 'gamesecond']
treatment_col = 'cm_dummy'


X = df[covariate_cols]
y = df[outcome_cols]
treatment = df[treatment_col].astype(bool).to_numpy()

# Define and Fit IPW Model.
learner = Pipeline([
    ('sclaer', preprocessing.MinMaxScaler()),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))
])
model = IPW(learner)
model.fit(X, treatment)

# metrics
print('F1 Score: ', metrics.f1_score(treatment, model.get_score(), threshold='auto'))
metrics.plot_roc_curve(treatment, model.get_score())
metrics.plot_probability_distribution(treatment, model.get_score())

# Estimate ATE
outcome_name = 'gamesecond'
z0, z1, treat_effect = model.estimate_effect(
    X, treatment, y[outcome_name].to_numpy().reshape(-1, 1), mode='ate')
metrics.plot_treatment_effect(outcome_name, z0[0], z1[0], treat_effect[0].round())
```

<div align="center">
    <h4> Effect size d. </h4>
    <img width="800px" height="300px" src="https://user-images.githubusercontent.com/17187586/97778595-981ffe80-1bbb-11eb-9f48-1fb1784bf17d.png">
    <h4> Propensity Score Distribution </h4>
    <img width="540px" height="480px" src="https://user-images.githubusercontent.com/17187586/97778619-bbe34480-1bbb-11eb-92d8-e00e25b2aa53.png">
    <h4> Average Treatment Effect </h4>
    <img width="420px" height="300px" src="https://user-images.githubusercontent.com/17187586/97778627-cac9f700-1bbb-11eb-94ea-e5ebd86a5270.png">
</div>

## Development

This project uses [uv](https://github.com/astral-sh/uv) for package management. To set up the development environment:

```shell
# Initialize the development environment
make init

# Run tests
make tests

# Run linting
make lint

# Build documentation
make docs
```

## Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

- [Konumaru](https://twitter.com/knmr_u) - Initial work

## Acknowledgments

- Uber / causalml: <https://github.com/uber/causalml>
- Iwanami Data Science Vol．3: <https://www.iwanami.co.jp/book/b243764.html>
