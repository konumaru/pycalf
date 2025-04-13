import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from pycalf.visualize import (
    plot_auuc,
    plot_effect_size,
    plot_lift_values,
    plot_probability_distribution,
    plot_roc_curve,
    plot_treatment_effect,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestVisualization:
    def test_plot_effect_size(self, sample_data):
        """Test plot_effect_size function"""
        X, _, treatment = sample_data

        # Test with default parameters
        ax = plot_effect_size(X, treatment)
        assert isinstance(ax, Axes)

        # Test with custom parameters
        ax = plot_effect_size(
            X,
            treatment,
            weight=np.ones(X.shape[0]),
            ascending=True,
            sortbyraw=False,
            figsize=(10, 5),
            threshold=0.2,
        )
        assert isinstance(ax, Axes)

    def test_plot_roc_curve(self):
        """Test plot_roc_curve function"""
        # Generate test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        # Test with default parameters
        ax = plot_roc_curve(y_true, y_score)
        assert isinstance(ax, Axes)

        # Test with custom figsize
        ax = plot_roc_curve(y_true, y_score, figsize=(8, 8))
        assert isinstance(ax, Axes)

    def test_plot_probability_distribution(self):
        """Test plot_probability_distribution function"""
        # Generate test data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        # Test with default parameters
        ax = plot_probability_distribution(y_true, y_score)
        assert isinstance(ax, Axes)

        # Test with custom figsize
        ax = plot_probability_distribution(y_true, y_score, figsize=(8, 8))
        assert isinstance(ax, Axes)

    def test_plot_treatment_effect(self):
        """Test plot_treatment_effect function"""
        # Test with default parameters
        ax = plot_treatment_effect(
            outcome_name="Test Outcome",
            control_effect=10,
            treat_effect=15,
            effect_size=5,
        )
        assert isinstance(ax, Axes)

        # Test with custom parameters
        ax = plot_treatment_effect(
            outcome_name="Test Outcome",
            control_effect=10,
            treat_effect=15,
            effect_size=5,
            figsize=(8, 6),
            fontsize=14,
        )
        assert isinstance(ax, Axes)

    def test_plot_auuc(self):
        """Test plot_auuc function"""
        # Generate test data
        uplift_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        lift = np.cumsum(np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]))
        baseline = np.array([0, 3, 6, 9, 12, 15, 18, 21, 25])

        # Test with default parameters
        ax = plot_auuc(uplift_score, lift, baseline)
        assert isinstance(ax, Axes)

        # Test with auuc parameter
        ax = plot_auuc(uplift_score, lift, baseline, auuc=0.75)
        assert isinstance(ax, Axes)

    def test_plot_lift_values(self):
        """Test plot_lift_values function"""
        # Generate test data
        labels = ["A", "B", "C", "D", "E"]
        values = np.array([10, 5, 8, 12, 3])

        # Test with default parameters
        ax = plot_lift_values(labels, values)
        assert isinstance(ax, Axes)

        # Test with custom figsize
        ax = plot_lift_values(labels, values, figsize=(10, 6))
        assert isinstance(ax, Axes)

        # Test with unequal lengths
        with pytest.raises(AssertionError):
            plot_lift_values(["A", "B"], [1, 2, 3])
