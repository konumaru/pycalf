from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pycalf.propensity import IPW, DoubleRobust, Matching


def test_ipw(sample_data) -> None:
    X, y, treatment = sample_data

    learner = LogisticRegression(solver="lbfgs", max_iter=200)
    ipw = IPW(learner=learner)
    ipw.fit(X, treatment, y)

    p_score = ipw.get_score()
    assert (p_score >= 0).all() and (
        p_score <= 1
    ).all(), "Probabilities are not within [0, 1]"

    ate_weight = ipw.get_weight(treatment)
    assert len(ate_weight) == len(X), "The number of samples is not equal"

    _ = ipw.estimate_effect(treatment, y)


def test_double_robust(sample_data) -> None:
    X, y, treatment = sample_data

    learner = LogisticRegression(solver="lbfgs", max_iter=200)
    second_learner = Pipeline(
        [
            ("sclaer", preprocessing.MinMaxScaler()),
            (
                "clf",
                LogisticRegression(solver="lbfgs", max_iter=200),
            ),
        ]
    )
    dr = DoubleRobust(learner=learner, second_learner=second_learner)
    dr.fit(X, treatment, y.to_numpy())

    p_score = dr.get_score()
    assert (p_score >= 0).all() and (
        p_score <= 1
    ).all(), "Probabilities are not within [0, 1]"

    ate_weight = dr.get_weight(treatment)
    assert len(ate_weight) == len(X), "The number of samples is not equal"

    _ = dr.estimate_effect(treatment)


def test_matching(sample_data) -> None:
    X, y, treatment = sample_data

    learner = LogisticRegression(solver="lbfgs", max_iter=200)
    matching = Matching(learner=learner)
    matching.fit(X, treatment, y)

    p_score = matching.get_score()
    assert (p_score >= 0).all() and (
        p_score <= 1
    ).all(), "Probabilities are not within [0, 1]"

    ate_weight = matching.get_weight(treatment)
    assert len(ate_weight) == len(X), "The number of samples is not equal"

    _ = matching.estimate_effect(treatment, y)
