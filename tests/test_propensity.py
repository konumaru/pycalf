import numpy as np
import pytest
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pycalf.propensity import IPW, DoubleRobust, Matching


class TestMatching:
    def test_initialization(self):
        """Test Matching initialization"""
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        matching = Matching(learner=learner)
        assert hasattr(matching, "learner")
        assert matching.learner == learner
        assert matching.min_match_dist == 1e-2

        # Test with custom min_match_dist
        matching = Matching(learner=learner, min_match_dist=0.05)
        assert matching.min_match_dist == 0.05

    def test_fit(self, sample_data):
        """Test Matching.fit method"""
        X, y, treatment = sample_data
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        matching = Matching(learner=learner)

        # Test fit with y
        matching.fit(X, treatment, y)
        assert hasattr(matching, "p_score")
        assert matching.p_score.shape[0] == X.shape[0]
        assert np.all(matching.p_score >= matching.eps)
        assert np.all(matching.p_score <= 1 - matching.eps)

        # Test fit without y
        matching = Matching(learner=learner)
        matching.fit(X, treatment)
        assert hasattr(matching, "p_score")
        assert matching.p_score.shape[0] == X.shape[0]

    def test_get_score(self, sample_data):
        """Test Matching.get_score method"""
        X, y, treatment = sample_data
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        matching = Matching(learner=learner)
        matching.fit(X, treatment)

        p_score = matching.get_score()
        assert p_score is not None
        assert p_score.shape[0] == X.shape[0]
        assert np.all(p_score >= matching.eps)
        assert np.all(p_score <= 1 - matching.eps)

    def test_get_weight(self, sample_data):
        """Test Matching.get_weight method"""
        X, y, treatment = sample_data
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        matching = Matching(learner=learner)
        matching.fit(X, treatment)

        # Test with raw mode
        raw_weight = matching.get_weight(treatment, mode="raw")
        assert raw_weight is not None
        assert raw_weight.shape[0] == X.shape[0]
        assert np.all(raw_weight == 1)

        # Test with ate mode
        ate_weight = matching.get_weight(treatment, mode="ate")
        assert ate_weight is not None
        assert ate_weight.shape[0] == X.shape[0]

        # Test with invalid mode
        with pytest.raises(
            AssertionError, match="mode must be string and it is raw　or ate."
        ):
            matching.get_weight(treatment, mode="invalid")

    def test_estimate_effect(self, sample_data):
        """Test Matching.estimate_effect method"""
        X, y, treatment = sample_data
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        matching = Matching(learner=learner)
        matching.fit(X, treatment)

        # Test with single outcome
        effect = matching.estimate_effect(treatment, y["gamecount"])
        assert isinstance(effect, tuple)
        assert len(effect) == 3

        # Test with different modes
        raw_effect = matching.estimate_effect(treatment, y["gamecount"], mode="raw")
        ate_effect = matching.estimate_effect(treatment, y["gamecount"], mode="ate")
        assert raw_effect != ate_effect  # Effects should differ between modes

        # Test with invalid mode
        with pytest.raises(AssertionError):
            matching.estimate_effect(treatment, y["gamecount"], mode="invalid")


class TestIPW:
    def test_initialization(self):
        """Test IPW initialization"""
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        ipw = IPW(learner=learner)
        assert hasattr(ipw, "learner")
        assert ipw.learner == learner
        assert ipw.p_score is None

    def test_fit(self, sample_data):
        """Test IPW.fit method"""
        X, y, treatment = sample_data
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        ipw = IPW(learner=learner)

        # Test fit with default eps
        ipw.fit(X, treatment)
        assert hasattr(ipw, "p_score")
        assert ipw.p_score.shape[0] == X.shape[0]
        assert np.all(ipw.p_score >= 1e-8)
        assert np.all(ipw.p_score <= 1 - 1e-8)

        # Test fit with custom eps
        ipw = IPW(learner=learner)
        ipw.fit(X, treatment, y, eps=0.01)
        assert hasattr(ipw, "p_score")
        assert np.all(ipw.p_score >= 0.01)
        assert np.all(ipw.p_score <= 0.99)

        # Test fit with invalid eps
        ipw = IPW(learner=learner)
        with pytest.raises(ValueError):
            ipw.fit(X, treatment, y, eps=1.0)

    def test_get_score(self, sample_data):
        """Test IPW.get_score method"""
        X, y, treatment = sample_data
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        ipw = IPW(learner=learner)
        ipw.fit(X, treatment)

        p_score = ipw.get_score()
        assert p_score is not None
        assert p_score.shape[0] == X.shape[0]
        assert np.all(p_score >= 1e-8)
        assert np.all(p_score <= 1 - 1e-8)

        # Test get_score without fit
        ipw = IPW(learner=learner)
        with pytest.raises(ValueError):
            ipw.get_score()

    def test_get_weight(self, sample_data):
        """Test IPW.get_weight method"""
        X, y, treatment = sample_data
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        ipw = IPW(learner=learner)
        ipw.fit(X, treatment)

        # Test with raw mode
        raw_weight = ipw.get_weight(treatment, mode="raw")
        assert raw_weight is not None
        assert raw_weight.shape[0] == X.shape[0]
        assert np.all(raw_weight == 1)

        # Test with ate mode
        ate_weight = ipw.get_weight(treatment, mode="ate")
        assert ate_weight is not None
        assert ate_weight.shape[0] == X.shape[0]

        # Test with att mode
        att_weight = ipw.get_weight(treatment, mode="att")
        assert att_weight is not None
        assert att_weight.shape[0] == X.shape[0]

        # Test with atu mode
        atu_weight = ipw.get_weight(treatment, mode="atu")
        assert atu_weight is not None
        assert atu_weight.shape[0] == X.shape[0]

        # Test with invalid mode
        with pytest.raises(
            AssertionError,
            match="mode must be string and it is must be raw, ate, att or atu.",
        ):
            ipw.get_weight(treatment, mode="invalid")

        # Test without fit
        ipw = IPW(learner=learner)
        with pytest.raises(ValueError):
            ipw.get_weight(treatment)

    def test_estimate_effect(self, sample_data):
        """Test IPW.estimate_effect method"""
        X, y, treatment = sample_data
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        ipw = IPW(learner=learner)
        ipw.fit(X, treatment)

        # Test with single outcome
        effect = ipw.estimate_effect(treatment, y["gamecount"])
        assert isinstance(effect, tuple)
        assert len(effect) == 3

        # Test with different modes
        raw_effect = ipw.estimate_effect(treatment, y["gamecount"], mode="raw")
        ate_effect = ipw.estimate_effect(treatment, y["gamecount"], mode="ate")
        att_effect = ipw.estimate_effect(treatment, y["gamecount"], mode="att")
        atu_effect = ipw.estimate_effect(treatment, y["gamecount"], mode="atu")

        # Effects should differ between modes
        assert raw_effect != ate_effect

        # Test with invalid mode
        with pytest.raises(AssertionError):
            ipw.estimate_effect(treatment, y["gamecount"], mode="invalid")


class TestDoubleRobust:
    def test_initialization(self):
        """Test DoubleRobust initialization"""
        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        second_learner = Pipeline(
            [
                ("scaler", preprocessing.MinMaxScaler()),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=200)),
            ]
        )
        dr = DoubleRobust(learner=learner, second_learner=second_learner)

        assert hasattr(dr, "learner")
        assert dr.learner == learner
        assert hasattr(dr, "treat_learner")
        assert hasattr(dr, "control_learner")
        assert dr.p_score is None

    def test_fit(self, sample_data):
        """Test DoubleRobust.fit method"""
        X, y, treatment = sample_data

        # 単一次元の配列を使用してテスト
        y_single = y["gamecount"].values  # numpyの1次元配列として取得

        # デバッグ用に情報を出力
        print(f"y_single shape: {y_single.shape}")
        print(f"X shape: {X.shape}")
        print(f"treatment shape: {treatment.shape}")

        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        second_learner = Pipeline(
            [
                ("scaler", preprocessing.MinMaxScaler()),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=200)),
            ]
        )
        dr = DoubleRobust(learner=learner, second_learner=second_learner)

        # トレースバックを出力してエラーの詳細を確認
        try:
            # 単一の結果変数でテスト（1次元配列として）
            dr.fit(X, treatment, y_single)
            assert hasattr(dr, "p_score")
            assert dr.p_score.shape[0] == X.shape[0]
            assert hasattr(dr, "y_control")
            assert hasattr(dr, "y_treat")
            # 形状をチェック：y_single.shapeは1次元なので直接比較できない
            assert dr.y_control.shape[0] == y_single.shape[0]
            assert dr.y_treat.shape[0] == y_single.shape[0]
        except Exception as e:
            import traceback

            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
            raise

        # Test fit with invalid eps
        dr = DoubleRobust(learner=learner, second_learner=second_learner)
        with pytest.raises(ValueError):
            dr.fit(X, treatment, y_single, eps=1.0)

    def test_estimate_effect(self, sample_data):
        """Test DoubleRobust.estimate_effect method"""
        X, y, treatment = sample_data
        # 単一の結果変数だけをテストに使用
        y_single = y["gamecount"].to_numpy().reshape(-1, 1)

        learner = LogisticRegression(solver="lbfgs", max_iter=200)
        second_learner = Pipeline(
            [
                ("scaler", preprocessing.MinMaxScaler()),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=200)),
            ]
        )
        dr = DoubleRobust(learner=learner, second_learner=second_learner)
        dr.fit(X, treatment, y_single)

        # Test with different modes
        raw_effect = dr.estimate_effect(treatment, mode="raw")
        ate_effect = dr.estimate_effect(treatment, mode="ate")
        att_effect = dr.estimate_effect(treatment, mode="att")
        atu_effect = dr.estimate_effect(treatment, mode="atu")

        assert isinstance(raw_effect, tuple)
        assert len(raw_effect) == 3

        # Effects should differ between modes
        assert raw_effect != ate_effect

        # Test with invalid mode
        with pytest.raises(AssertionError):
            dr.estimate_effect(treatment, mode="invalid")

        # Test without fit
        dr = DoubleRobust(learner=learner, second_learner=second_learner)
        with pytest.raises(ValueError):
            dr.estimate_effect(treatment)
