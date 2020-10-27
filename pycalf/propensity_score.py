import numpy as np
import pandas as pd


class IPW():
    """Inverse Probability Weighting Method.
    """

    def __init__(self, learner):
        self.learner = learner
        self.p_score = None

    def fit(self, X, treatment, clip=1e-15):
        """Fit Leaner and Calculation IPW.

        Parameters
        ----------
        X : numpy ndarray, DataFrame

        treatment : numpy ndarray, Series

        clip : float

        Returns
        -------
        None
        """
        self.learner.fit(X, treatment)
        assert 0 <= clip < 1, 'clip must be 0 to 1.'
        self.p_score = np.clip(self.learner.predict_proba(X)[:, 1], clip, 1 - clip)

    def get_score(self):
        return self.p_score

    def get_weight(self, treatment, mode='ate'):
        self._check_mode(mode)
        if mode == 'raw':
            return np.ones(treatment.shape[0])
        elif mode == 'ate':
            return np.where(treatment == 1, 1 / self.p_score, 1 / (1 - self.p_score))
        elif mode == 'att':
            return np.where(treatment == 1, 1, self.p_score / (1 - self.p_score))
        elif mode == 'atu':
            return np.where(treatment == 1, (1 - self.p_score) / self.p_score, 1)

    def estimate_effect(self, treatment, y, mode='ate'):
        """Description

        Parameters
        ----------
        treatment : np.ndarray, pd.Series

        outcomes : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        self._check_mode(mode)
        weight = self.get_weight(treatment, mode=mode)
        return self._estimate_outcomes(treatment, y, weight)

    def _check_mode(self, mode):
        mode_list = ['raw', 'ate', 'att', 'atu']
        assert mode in mode_list, 'mode must be string and it is raw, ate, att or atu.'

    def _estimate_outcomes(self, treatment, y, weight):
        """Description

        Parameters
        ----------
        treatment : np.ndarray

        y :  np.ndarray

        weight : np.ndarray

        Returns
        -------
        (y_control, y_treat, effect_size) : tuple
        """
        y_control = np.average(y[~treatment], axis=0, weights=weight[~treatment])
        y_treat = np.average(y[treatment], axis=0, weights=weight[treatment])
        effect_size = y_treat - y_control
        return (y_control, y_treat, effect_size)
