import numpy as np
import pandas as pd
import pytest

from pycalf.metrics import VIF, AttributeEffect, EffectSize, f1_score


class TestEffectSize:
    def test_initialization(self):
        """Test EffectSize initialization"""
        es = EffectSize()
        assert es.effect_size is None
        assert es.effect_name is None

    def test_fit(self, sample_data):
        """Test EffectSize.fit method"""
        X, _, treatment = sample_data

        # Test with default weight
        es = EffectSize()
        es.fit(X, treatment)
        assert es.effect_size is not None
        assert es.effect_name is not None
        assert len(es.effect_size) == X.shape[1]
        assert len(es.effect_name) == X.shape[1]

        # Test with custom weight
        weight = np.ones(X.shape[0])
        weight[:10] = 2.0  # Give more weight to first 10 samples
        es = EffectSize()
        es.fit(X, treatment, weight=weight)
        assert es.effect_size is not None
        assert es.effect_name is not None

    def test_transform(self, sample_data):
        """Test EffectSize.transform method"""
        X, _, treatment = sample_data

        es = EffectSize()
        es.fit(X, treatment)
        result = es.transform()

        assert isinstance(result, dict)
        assert "effect_name" in result
        assert "effect_size" in result
        assert isinstance(result["effect_name"], np.ndarray)
        assert isinstance(result["effect_size"], np.ndarray)

        # Test transform without fit
        es = EffectSize()
        with pytest.raises(ValueError):
            es.transform()

    def test_fit_transform(self, sample_data):
        """Test EffectSize.fit_transform method"""
        X, _, treatment = sample_data

        es = EffectSize()
        result = es.fit_transform(X, treatment)

        assert isinstance(result, dict)
        assert "effect_name" in result
        assert "effect_size" in result
        assert isinstance(result["effect_name"], np.ndarray)
        assert isinstance(result["effect_size"], np.ndarray)


class TestAttributeEffect:
    def test_initialization(self):
        """Test AttributeEffect initialization"""
        ae = AttributeEffect()
        assert not hasattr(ae, "result") or ae.result is None
        assert ae.treat_result is None
        assert ae.control_result is None

    def test_fit(self, sample_data):
        """Test AttributeEffect.fit method"""
        X, y, treatment = sample_data

        # Test with single outcome
        ae = AttributeEffect()
        ae.fit(X, treatment, y["gamecount"])
        assert ae.treat_result is not None
        assert ae.control_result is not None

        # Test with custom weight
        weight = np.ones(X.shape[0])
        weight[:10] = 2.0  # Give more weight to first 10 samples
        ae = AttributeEffect()
        ae.fit(X, treatment, y["gamecount"], weight=weight)
        assert ae.treat_result is not None
        assert ae.control_result is not None

    def test_transform(self, sample_data):
        """Test AttributeEffect.transform method"""
        X, y, treatment = sample_data

        ae = AttributeEffect()
        ae.fit(X, treatment, y["gamecount"])
        result = ae.transform()

        assert isinstance(result, pd.DataFrame)
        assert "Z0_effect" in result.columns
        assert "Z1_effect" in result.columns
        assert "Z0_tvalue" in result.columns
        assert "Z1_tvalue" in result.columns
        assert "Lift" in result.columns

        # Test transform without fit
        ae = AttributeEffect()
        with pytest.raises(ValueError):
            ae.transform()


class TestVIF:
    def test_initialization(self):
        """Test VIF initialization"""
        vif = VIF()
        assert not hasattr(vif, "result") or vif.result is None

    def test_fit(self, sample_data):
        """Test VIF.fit method"""
        X, _, _ = sample_data

        vif = VIF()
        vif.fit(X)
        assert hasattr(vif, "result")
        assert isinstance(vif.result, pd.DataFrame)
        assert vif.result.shape[0] == X.shape[1]

    def test_transform(self, sample_data):
        """Test VIF.transform method"""
        X, _, _ = sample_data

        vif = VIF()
        vif.fit(X)
        result = vif.transform()

        assert isinstance(result, pd.DataFrame)
        assert "VIF" in result.columns
        assert result.shape[0] == X.shape[1]

    def test_fit_transform(self, sample_data):
        """Test VIF.fit_transform method"""
        X, _, _ = sample_data

        vif = VIF()
        result = vif.fit_transform(X)

        assert isinstance(result, pd.DataFrame)
        assert "VIF" in result.columns
        assert result.shape[0] == X.shape[1]


def test_f1_score():
    """Test f1_score function"""
    # Test with perfect prediction
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    score = f1_score(y_true, y_score, threshold=0.5, is_auto=False)
    assert score == 1.0

    # Test with imperfect prediction
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.6, 0.2, 0.3, 0.8])
    score = f1_score(y_true, y_score, threshold=0.5, is_auto=False)
    assert 0 <= score <= 1.0

    # Test with auto threshold
    score_auto = f1_score(y_true, y_score, is_auto=True)
    assert 0 <= score_auto <= 1.0

    # Test threshold validation
    with pytest.raises(AssertionError):
        f1_score(y_true, y_score, threshold=1.0, is_auto=False)
