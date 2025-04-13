import numpy as np
from sklearn.linear_model import LogisticRegression

from pycalf.uplift import UpliftModel


class TestUpliftModel:
    def test_initialization(self):
        """Test UpliftModel initialization"""
        learner_treat = LogisticRegression(solver="lbfgs", max_iter=200)
        learner_control = LogisticRegression(solver="lbfgs", max_iter=200)
        model = UpliftModel(learner_treat, learner_control)

        assert hasattr(model, "learner_treat")
        assert model.learner_treat == learner_treat
        assert hasattr(model, "learner_control")
        assert model.learner_control == learner_control

    def test_fit(self, sample_data):
        """Test UpliftModel.fit method"""
        X, y, treatment = sample_data
        # Prepare data for uplift model
        X_treat = X[treatment].copy()
        y_treat = y["gamecount"][treatment].copy()
        X_control = X[~treatment].copy()
        y_control = y["gamecount"][~treatment].copy()

        # Create model
        learner_treat = LogisticRegression(solver="lbfgs", max_iter=200)
        learner_control = LogisticRegression(solver="lbfgs", max_iter=200)
        model = UpliftModel(learner_treat, learner_control)

        # Test fit with default weights
        model.fit(
            X_treat,
            y_treat > y_treat.median(),
            X_control,
            y_control > y_control.median(),
        )
        assert hasattr(model.learner_treat, "coef_")
        assert hasattr(model.learner_control, "coef_")

        # Test fit with custom weights
        weight_treat = np.ones(X_treat.shape[0])
        weight_control = np.ones(X_control.shape[0])
        model = UpliftModel(learner_treat, learner_control)
        model.fit(
            X_treat,
            y_treat > y_treat.median(),
            X_control,
            y_control > y_control.median(),
            weight_treat,
            weight_control,
        )
        assert hasattr(model.learner_treat, "coef_")
        assert hasattr(model.learner_control, "coef_")

    def test_estimate_uplift_score(self, sample_data):
        """Test UpliftModel.estimate_uplift_score method"""
        X, y, treatment = sample_data
        # Prepare binary outcome data for uplift model
        X_treat = X[treatment].copy()
        y_treat = y["gamedummy"][treatment].copy()
        X_control = X[~treatment].copy()
        y_control = y["gamedummy"][~treatment].copy()

        # Create and fit model
        learner_treat = LogisticRegression(solver="lbfgs", max_iter=200)
        learner_control = LogisticRegression(solver="lbfgs", max_iter=200)
        model = UpliftModel(learner_treat, learner_control)
        model.fit(X_treat, y_treat, X_control, y_control)

        # Test estimating uplift scores
        uplift_scores = model.estimate_uplift_score(X)
        assert uplift_scores is not None
        assert uplift_scores.shape[0] == X.shape[0]

    def test_predict(self, sample_data):
        """Test UpliftModel.predict method"""
        X, y, treatment = sample_data
        # Prepare binary outcome data for uplift model
        X_treat = X[treatment].copy()
        y_treat = y["gamedummy"][treatment].copy()
        X_control = X[~treatment].copy()
        y_control = y["gamedummy"][~treatment].copy()

        # Create and fit model
        learner_treat = LogisticRegression(solver="lbfgs", max_iter=200)
        learner_control = LogisticRegression(solver="lbfgs", max_iter=200)
        model = UpliftModel(learner_treat, learner_control)
        model.fit(X_treat, y_treat, X_control, y_control)

        # Test predict
        uplift_score, lift = model.predict(X, treatment, y["gamedummy"].to_numpy())
        assert uplift_score is not None
        assert lift is not None
        assert uplift_score.shape[0] == X.shape[0]
        assert lift.shape[0] == X.shape[0]

    def test_get_baseline(self, sample_data):
        """Test UpliftModel.get_baseline method"""
        X, y, treatment = sample_data
        # Prepare binary outcome data for uplift model
        X_treat = X[treatment].copy()
        y_treat = y["gamedummy"][treatment].copy()
        X_control = X[~treatment].copy()
        y_control = y["gamedummy"][~treatment].copy()

        # Create and fit model
        learner_treat = LogisticRegression(solver="lbfgs", max_iter=200)
        learner_control = LogisticRegression(solver="lbfgs", max_iter=200)
        model = UpliftModel(learner_treat, learner_control)
        model.fit(X_treat, y_treat, X_control, y_control)

        # Get uplift predictions
        _, lift = model.predict(X, treatment, y["gamedummy"].to_numpy())

        # Test get_baseline
        baseline = model.get_baseline(lift)
        assert baseline is not None
        assert baseline.shape == lift.shape
        assert baseline[0] == 0  # First value should be 0
        assert np.isclose(baseline[-1], lift[-1])  # Last values should be equal

    def test_get_auuc(self, sample_data):
        """Test UpliftModel.get_auuc method"""
        X, y, treatment = sample_data
        # Prepare binary outcome data for uplift model
        X_treat = X[treatment].copy()
        y_treat = y["gamedummy"][treatment].copy()
        X_control = X[~treatment].copy()
        y_control = y["gamedummy"][~treatment].copy()

        # Create and fit model
        learner_treat = LogisticRegression(solver="lbfgs", max_iter=200)
        learner_control = LogisticRegression(solver="lbfgs", max_iter=200)
        model = UpliftModel(learner_treat, learner_control)
        model.fit(X_treat, y_treat, X_control, y_control)

        # Get uplift predictions
        _, lift = model.predict(X, treatment, y["gamedummy"].to_numpy())

        # Test get_auuc
        auuc = model.get_auuc(lift)
        assert auuc is not None
        assert isinstance(auuc, float)
