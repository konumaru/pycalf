import copy

import numpy as np
from sklearn.neighbors import NearestNeighbors


class Matching:
    """
    Matching with propensity score.

    Attributes
    ----------
    p_score : numpy.ndarray
        Propensity Score.
    """

    def __init__(self, learner, min_match_dist=1e-2) -> None:
        """
        Parameters
        ----------
        learner :
            Learner to estimate propensity score.
        """
        self.p_score: np.ndarray
        self.eps: float = 1e-8

        self.learner = learner
        self.min_match_dist = min_match_dist

    def fit(self, X, treatment, y) -> None:
        """
        Fit learner and Estimate Propensity Score.

        Parameters
        ----------
        X : numpy.ndarray
            Covariates for propensity score.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        esp : float
            Extreme Value Trend Score Rounding Value.
        """
        self.learner.fit(X, treatment)
        self.p_score = np.clip(
            self.learner.predict_proba(X)[:, 1], self.eps, 1 - self.eps
        )

    def get_score(self):
        """
        Return propensity score.
        """
        return self.p_score

    def get_weight(self, treatment, mode="ate"):
        """
        Return sample weight representing matching.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        mode : str
            Adjustment method. raw or ate.

        Returns
        -------
        sampel_weight : numpy.ndarray
        """
        self._check_mode(mode)
        if mode == "raw":
            return np.ones(treatment.shape[0])
        elif mode == "ate":
            return self._get_matching_weight(treatment)
        else:
            raise ValueError("mode must be raw or ate.")

    def _check_mode(self, mode):
        """
        Check if it is a supported mode.

        Parameters
        ----------
        mode : str
            Adjustment method. raw or ate.
        """
        mode_list = ["raw", "ate"]
        assert mode in mode_list, "mode must be string and it is raw　or ate."

    def _get_matching_weight(self, treatment):
        """
        Match using propensity score and return sample_weight.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.

        Returns
        -------
        sampel_weight : numpy.ndarray
        """
        score = self.p_score
        treat_idx, control_idx = self._get_nearest_idx(treatment, score)

        matching_idx = np.concatenate((treat_idx, control_idx), axis=0)
        idx, counts = np.unique(matching_idx, return_counts=True)
        weight = np.zeros(treatment.shape[0])
        weight[idx] = counts
        return weight

    def _get_nearest_idx(self, treatment, score):
        """
        Match the closest data between groups with and without intervention.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        score : numpy.ndarray
            Propensity Score.

        Returns
        -------
        treat_idx : numpy.ndarray
            Sample index of treatment group.
        control_idx : numpy.ndarray
            Sample index of control group.
        """
        score = score.reshape(-1, 1)
        control_size, treat_size = (~treatment).sum(), treatment.sum()
        major_sample_group = np.argmax([control_size, treat_size])

        neigh = NearestNeighbors(n_neighbors=5, metric="manhattan")
        neigh.fit(score[treatment == major_sample_group])
        distance, match_idx = neigh.kneighbors(
            score[treatment != major_sample_group], 1, return_distance=True
        )
        match_idx = match_idx[distance < self.min_match_dist].flatten()

        if major_sample_group == 1:
            treat_idx = np.where(treatment)[0][match_idx]
            control_idx = np.where(~treatment)[0]
        else:
            treat_idx = np.where(treatment)[0]
            control_idx = np.where(~treatment)[0][match_idx]
        return treat_idx, control_idx

    def estimate_effect(self, treatment, y, mode="ate"):
        """
        Match using propensity score and return sample_weight.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        mode : str
            Adjustment method. raw or ate.

        Returns
        -------
        ajusted_outcomes : numpy.ndarray
        """
        self._check_mode(mode)
        weight = self.get_weight(treatment, mode=mode)
        return self._estimate_outcomes(treatment, y, weight)

    def _estimate_outcomes(self, treatment, y, weight):
        """
        Match using propensity score and return sample_weight.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        weight : numpy.ndarray
            sample weight with matching.

        Returns
        -------
        avg_y_control : float
            average outcome of control group.
        avg_y_treat : float
            average outcome of treatment group.
        effect_size : float
            diff of average_y_treatment and average_y_control
        """
        avg_y_control = np.average(
            y[~treatment], axis=0, weights=weight[~treatment]
        )
        avg_y_treat = np.average(
            y[treatment], axis=0, weights=weight[treatment]
        )
        effect_size = avg_y_treat - avg_y_control
        return (avg_y_control, avg_y_treat, effect_size)


class IPW:
    """Inverse Probability Weighting Method."""

    def __init__(self, learner) -> None:
        """
        Parameters
        ----------
        learner :
            Learner to estimate propensity score.
        """
        self.p_score: np.ndarray

        self.learner = learner

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        """
        Fit learner and Estimate Propensity Score.

        Parameters
        ----------
        X : numpy.ndarray
            Covariates for propensity score.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        esp : float
            Extreme Value Trend Score Rounding Value.
        """
        self.learner.fit(X, treatment)
        assert 0 <= eps < 1, "clip must be 0 to 1."
        pred = self.learner.predict_proba(X)[:, 1]
        self.p_score = np.clip(pred, eps, 1 - eps)

    def get_score(self) -> np.ndarray:
        """
        Return propensity score.
        """
        return self.p_score

    def get_weight(self, treatment, mode="ate"):
        """
        Return sample weight representing matching.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        mode : str
            Adjustment method. must be raw, ate, att or atu.

        Returns
        -------
        sampel_weight : numpy.ndarray
        """
        self._check_mode(mode)
        if mode == "raw":
            return np.ones(treatment.shape[0])
        elif mode == "ate":
            return np.where(
                treatment == 1, 1 / self.p_score, 1 / (1 - self.p_score)
            )
        elif mode == "att":
            return np.where(
                treatment == 1, 1, self.p_score / (1 - self.p_score)
            )
        elif mode == "atu":
            return np.where(
                treatment == 1, (1 - self.p_score) / self.p_score, 1
            )
        else:
            raise ValueError("mode must be raw, ate, att or atu.")

    def estimate_effect(self, treatment, y, mode="ate"):
        """
        Match using propensity score and return sample_weight.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        mode : str
            Adjustment method. must be raw, ate, att or atu.

        Returns
        -------
        ajusted_outcomes : numpy.ndarray
        """
        self._check_mode(mode)
        weight = self.get_weight(treatment, mode=mode)
        return self._estimate_outcomes(treatment, y, weight)

    def _check_mode(self, mode):
        """
        Check if it is a supported mode.

        Parameters
        ----------
        mode : str
            Adjustment method. must be raw, ate, att or atu.
        """
        mode_list = ["raw", "ate", "att", "atu"]
        assert (
            mode in mode_list
        ), "mode must be string and it is must be raw, ate, att or atu."

    def _estimate_outcomes(self, treatment, y, weight):
        """
        Match using propensity score and return sample_weight.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        weight : numpy.ndarray
            sample weight with ipw.

        Returns
        -------
        avg_y_control : float
            average outcome of control group.
        avg_y_treat : float
            average outcome of treatment group.
        effect_size : float
            diff of average_y_treatment and average_y_control
        """
        avg_y_control = np.average(
            y[~treatment], axis=0, weights=weight[~treatment]
        )
        avg_y_treat = np.average(
            y[treatment], axis=0, weights=weight[treatment]
        )
        effect_size = avg_y_treat - avg_y_control
        return (avg_y_control, avg_y_treat, effect_size)


class DoubleRobust(IPW):
    def __init__(self, learner, second_learner) -> None:
        """
        Parameters
        ----------
        learner :
            Learner to estimate propensity score.
        second_learner :
            Learner to estimate anti-real virtual intervention effect.
        """
        super(DoubleRobust, self).__init__(learner)
        self.treat_learner = copy.deepcopy(second_learner)
        self.control_learner = copy.deepcopy(second_learner)

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        """
        Fit learner and Estimate Propensity Score.

        Parameters
        ----------
        X : numpy.ndarray
            Covariates for propensity score.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        esp : float
            Extreme Value Trend Score Rounding Value.
        """
        self.learner.fit(X, treatment)
        assert 0 <= eps < 1, "clip must be 0 to 1."
        self.p_score = np.clip(
            self.learner.predict_proba(X)[:, 1], eps, 1 - eps
        )

        self.y_control = np.zeros(y.shape)
        self.y_treat = np.zeros(y.shape)
        # Fit second models
        for i, _y in enumerate(y.T):
            self.treat_learner.fit(X[treatment], _y[treatment])
            self.control_learner.fit(X[~treatment], _y[~treatment])

            self.y_control[:, i] = np.where(
                ~treatment, _y, self.control_learner.predict(X)
            )
            self.y_treat[:, i] = np.where(
                treatment, _y, self.treat_learner.predict(X)
            )

    def estimate_effect(self, treatment, mode="ate"):
        """
        Match using propensity score and return sample_weight.

        Parameters
        ----------
        X : numpy.ndarray
            Covariates for propensity score.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        mode : str
            Adjustment method. must be raw, ate, att or atu.

        Returns
        -------
        ajusted_outcomes : numpy.ndarray
        """
        self._check_mode(mode)
        weight = self.get_weight(treatment, mode=mode)
        return self._estimate_outcomes(weight)

    def _estimate_outcomes(self, weight):
        """
        Match using propensity score and return sample_weight.

        Parameters
        ----------
        X : numpy.ndarray
            Covariates for propensity score.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        weight : numpy.ndarray
            sample weight of ipw.

        Returns
        -------
        avg_y_control : float
            average outcome of control group.
        avg_y_treat : float
            average outcome of treatment group.
        effect_size : float
            diff of average_y_treatment and average_y_control
        """
        avg_y_control = np.average(self.y_control, axis=0, weights=weight)
        avg_y_treat = np.average(self.y_treat, axis=0, weights=weight)
        effect_size = avg_y_treat - avg_y_control
        return (avg_y_control, avg_y_treat, effect_size)
