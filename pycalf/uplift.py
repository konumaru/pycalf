from typing import Optional, Tuple

import numpy as np


class UpliftModel:
    """Class of Uplift Modeling."""

    def __init__(self, learner_treat, learner_control) -> None:
        """
        Parameters
        ----------
        learner_treat : object
            Learner to estimate effect of treatment group. Must have fit and predict_proba methods.
        learner_control : object
            Learner to estimate effect of control group. Must have fit and predict_proba methods.
        """
        self.learner_treat = learner_treat
        self.learner_control = learner_control

    def fit(
        self,
        X_treat: np.ndarray,
        y_treat: np.ndarray,
        X_control: np.ndarray,
        y_control: np.ndarray,
        weight_treat: Optional[np.ndarray] = None,
        weight_control: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        X_treat : numpy.ndarray
            Features for learner_treat.
        y_treat : numpy.ndarray
            Labels for learner_treat.
        X_control : numpy.ndarray
            Features for learner_control.
        y_control : numpy.ndarray
            Labels for learner_control.
        weight_treat : numpy.ndarray or None
            Weights for learner_treat.
        weight_control : numpy.ndarray or None
            Weights for learner_control.

        Returns
        -------
        None
        """
        self.learner_treat.fit(X_treat, y_treat, sample_weight=weight_treat)
        self.learner_control.fit(X_control, y_control, sample_weight=weight_control)

    def estimate_uplift_score(self, X: np.ndarray) -> np.ndarray:
        """Estimate uplift scores.

        Parameters
        ----------
        X : numpy.ndarray
            Features for prediction treat and control probability.

        Returns
        -------
        uplift_score : np.array
            Uplift Score.
        """
        proba_treat = self.learner_treat.predict_proba(X)[:, 1]
        proba_control = self.learner_control.predict_proba(X)[:, 1]
        uplift_score = proba_treat / proba_control
        return uplift_score

    def predict(
        self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        X : numpy.ndarray
            Features for prediction treat and control probability.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.

        Returns
        -------
        (uplift_score, lift) : tuple
            Uplift score and lift values.
        """
        uplift_score = self.estimate_uplift_score(X)

        sorted_idx = np.argsort(uplift_score)[::-1]
        uplift_score = uplift_score[sorted_idx]
        y = y[sorted_idx]
        treatment = treatment[sorted_idx]

        y_treat = np.nancumsum(np.where(treatment == 1, y, np.nan))
        y_control = np.nancumsum(np.where(treatment == 0, y, np.nan))

        treat_size = np.nancumsum(np.where(treatment == 1, treatment, np.nan))
        control_size = np.nancumsum(np.where(treatment == 0, (1 - treatment), np.nan))

        cumavg_y_treat = np.array(
            [0.0 if s == 0 else _y / s for _y, s in zip(y_treat, treat_size)]
        )
        cumavg_y_control = np.array(
            [0.0 if s == 0 else _y / s for _y, s in zip(y_control, control_size)]
        )

        lift = (cumavg_y_treat - cumavg_y_control) * treat_size
        return (uplift_score, lift)

    def get_baseline(self, lift: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        lift : numpy.ndarray
            Array of lift, treatment effect.

        Returns
        -------
        base_line : numpy.ndarray
            Array of random treat effect.
        """
        data_size = len(lift)
        # Ensure that the first element is exactly 0 and last element matches lift[-1]
        base_line = np.linspace(0, lift[data_size - 1], data_size)
        return base_line

    def get_auuc(self, lift: np.ndarray) -> float:
        """
        Parameters
        ----------
        lift : numpy.ndarray
            Array of lift, treatment effect.

        Returns
        -------
        auuc : float
            AUUC score.
        """
        base_line = self.get_baseline(lift)
        auuc = (lift - base_line).sum() / len(lift)
        return auuc
