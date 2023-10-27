import numpy as np
import pandas as pd

from pycalf.metrics import VIF, AttributeEffect, EffectSize, f1_score


def test_vif(sample_data) -> None:
    X, y, treatment = sample_data

    vif = VIF()
    vif.fit(X)

    result = vif.transform()
    assert isinstance(result, pd.DataFrame), "The result is not DataFrame."

    result = vif.fit_transform(X)
    assert isinstance(result, pd.DataFrame), "The result is not DataFrame."


def test_attribute_effect(sample_data) -> None:
    X, y, treatment = sample_data

    ae = AttributeEffect()
    ae.fit(X, treatment, y["gamecount"])

    result = ae.transform()
    assert isinstance(result, pd.DataFrame), "The result is not DataFrame."


def test_effect_size(sample_data) -> None:
    X, y, treatment = sample_data

    es = EffectSize()
    es.fit(X, treatment, y["gamecount"])

    result = es.transform()
    assert isinstance(result, tuple), "The result is not tuple."

    result = es.fit_transform(X, treatment, y["gamecount"])
    assert isinstance(result, tuple), "The result is not tuple."


def test_f1_score() -> None:
    preds = np.random.rand(10)
    labels = (preds > 0.5).astype(int)

    score = f1_score(labels, preds)
    assert isinstance(score, float), "The score is not float."
