import copy
from typing import Optional

import numpy as np
import pandas as pd
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
        learner : object
            Learner to estimate propensity score. Must have fit and predict_proba methods.
        min_match_dist : float, default=1e-2
            Minimum distance for matching.
        """
        self.p_score: np.ndarray
        self.eps: float = 1e-8

        self.learner = learner
        self.min_match_dist = min_match_dist

    def fit(
        self, X: pd.DataFrame, treatment: np.ndarray, y: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit learner and Estimate Propensity Score.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates for propensity score.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        """
        self.learner.fit(X, treatment)
        self.p_score = np.clip(
            self.learner.predict_proba(X)[:, 1], self.eps, 1 - self.eps
        )

    def get_score(self) -> np.ndarray:
        """
        Return propensity score.

        Returns
        -------
        numpy.ndarray
            Propensity score.
        """
        return self.p_score

    def get_weight(self, treatment: np.ndarray, mode: str = "ate") -> np.ndarray:
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
        numpy.ndarray
            Sample weight.
        """
        self._check_mode(mode)
        if mode == "raw":
            return np.ones(treatment.shape[0])
        elif mode == "ate":
            return self._get_matching_weight(treatment)
        else:
            raise ValueError("mode must be raw or ate.")

    def _check_mode(self, mode: str) -> None:
        """
        Check if it is a supported mode.

        Parameters
        ----------
        mode : str
            Adjustment method. raw or ate.
        """
        mode_list = ["raw", "ate"]
        assert mode in mode_list, "mode must be string and it is raw　or ate."

    def _get_matching_weight(self, treatment: np.ndarray) -> np.ndarray:
        """
        Match using propensity score and return sample_weight.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.

        Returns
        -------
        numpy.ndarray
            Sample weight.
        """
        score = self.p_score
        treat_idx, control_idx = self._get_nearest_idx(treatment, score)

        matching_idx = np.concatenate((treat_idx, control_idx), axis=0)
        idx, counts = np.unique(matching_idx, return_counts=True)
        weight = np.zeros(treatment.shape[0])
        weight[idx] = counts
        return weight

    def _get_nearest_idx(
        self, treatment: np.ndarray, score: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
        # Flatten distance array before comparison
        distance = distance.flatten()
        match_idx = match_idx[distance < self.min_match_dist].flatten()

        if major_sample_group == 1:
            treat_idx = np.where(treatment)[0][match_idx]
            control_idx = np.where(~treatment)[0]
        else:
            treat_idx = np.where(treatment)[0]
            control_idx = np.where(~treatment)[0][match_idx]
        return treat_idx, control_idx

    def estimate_effect(
        self, treatment: np.ndarray, y: np.ndarray, mode: str = "ate"
    ) -> tuple[float, float, float]:
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
        tuple
            A tuple containing (avg_y_control, avg_y_treat, effect_size)
        """
        self._check_mode(mode)
        weight = self.get_weight(treatment, mode=mode)
        return self._estimate_outcomes(treatment, y, weight)

    def _estimate_outcomes(
        self, treatment: np.ndarray, y: np.ndarray, weight: np.ndarray
    ) -> tuple[float, float, float]:
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
        tuple
            A tuple containing (avg_y_control, avg_y_treat, effect_size)
        """
        avg_y_control = np.average(y[~treatment], axis=0, weights=weight[~treatment])
        avg_y_treat = np.average(y[treatment], axis=0, weights=weight[treatment])
        effect_size = avg_y_treat - avg_y_control
        return (avg_y_control, avg_y_treat, effect_size)


class IPW:
    """Inverse Probability Weighting Method."""

    def __init__(self, learner) -> None:
        """
        Parameters
        ----------
        learner : object
            Learner to estimate propensity score. Must have fit and predict_proba methods.
        """
        self.p_score: np.ndarray = None  # type: ignore

        self.learner = learner

    def fit(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: Optional[np.ndarray] = None,
        eps: float = 1e-8,
    ) -> None:
        """
        Fit learner and Estimate Propensity Score.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates for propensity score.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables.
        eps : float, default=1e-8
            Extreme Value Trend Score Rounding Value.

        Raises
        ------
        ValueError
            If eps is not in range [0, 1).
        """
        if not 0 <= eps < 1:
            raise ValueError("eps must be in range [0, 1).")

        self.learner.fit(X, treatment)
        pred = self.learner.predict_proba(X)[:, 1]
        self.p_score = np.clip(pred, eps, 1 - eps)

    def get_score(self) -> np.ndarray:
        """
        Return propensity score.

        Returns
        -------
        p_score : numpy.ndarray
            Propensity score for each sample.

        Raises
        ------
        ValueError
            If model is not fitted.
        """
        if not hasattr(self, "p_score") or self.p_score is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.p_score

    def get_weight(self, treatment: np.ndarray, mode: str = "ate") -> np.ndarray:
        """
        Return sample weight representing matching.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        mode : str, default="ate"
            Adjustment method. Must be 'raw', 'ate', 'att' or 'atu'.

        Returns
        -------
        sample_weight : numpy.ndarray
            Sample weights.

        Raises
        ------
        ValueError
            If mode is not 'raw', 'ate', 'att' or 'atu'.
        """
        self._check_mode(mode)
        if not hasattr(self, "p_score") or self.p_score is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if mode == "raw":
            return np.ones(treatment.shape[0])
        elif mode == "ate":
            return np.where(treatment == 1, 1 / self.p_score, 1 / (1 - self.p_score))
        elif mode == "att":
            return np.where(treatment == 1, 1, self.p_score / (1 - self.p_score))
        elif mode == "atu":
            return np.where(treatment == 1, (1 - self.p_score) / self.p_score, 1)
        else:
            raise ValueError("mode must be raw, ate, att or atu.")

    def estimate_effect(
        self, treatment: np.ndarray, y: np.ndarray, mode: str = "ate"
    ) -> tuple[float, float, float]:
        """
        Calculate treatment effect using inverse probability weighting.

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
        tuple
            A tuple containing (avg_y_control, avg_y_treat, effect_size)
        """
        self._check_mode(mode)
        weight = self.get_weight(treatment, mode=mode)
        return self._estimate_outcomes(treatment, y, weight)

    def _check_mode(self, mode: str) -> None:
        """
        Check if it is a supported mode.

        Parameters
        ----------
        mode : str
            Adjustment method. must be raw, ate, att or atu.
        """
        mode_list = ["raw", "ate", "att", "atu"]
        assert mode in mode_list, (
            "mode must be string and it is must be raw, ate, att or atu."
        )

    def _estimate_outcomes(
        self, treatment: np.ndarray, y: np.ndarray, weight: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Calculate treatment effect using provided weights.

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
        tuple
            A tuple containing (avg_y_control, avg_y_treat, effect_size)
        """
        avg_y_control = np.average(y[~treatment], axis=0, weights=weight[~treatment])
        avg_y_treat = np.average(y[treatment], axis=0, weights=weight[treatment])
        effect_size = avg_y_treat - avg_y_control
        return (avg_y_control, avg_y_treat, effect_size)


class DoubleRobust(IPW):
    def __init__(self, learner, second_learner) -> None:
        """
        Parameters
        ----------
        learner : object
            Learner to estimate propensity score. Must have fit and predict_proba methods.
        second_learner : object
            Learner to estimate anti-real virtual intervention effect. Must have fit and predict methods.
        """
        super(DoubleRobust, self).__init__(learner)
        self.treat_learner = copy.deepcopy(second_learner)
        self.control_learner = copy.deepcopy(second_learner)

    def fit(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        """
        Fit learner and Estimate Propensity Score.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates for propensity score.
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        y : numpy.ndarray
            Outcome variables. Can be 1D or 2D array.
        eps : float, default=1e-8
            Extreme Value Trend Score Rounding Value.

        Raises
        ------
        ValueError
            If eps is not in range [0, 1).
        """
        # 型を明示的にブール値に変換
        treatment = treatment.astype(bool)

        # DataFrameのコピーを作成し操作する（参照渡しによる問題を回避）
        X_df = X.copy()

        self.learner.fit(X_df, treatment)
        if not 0 <= eps < 1:
            raise ValueError("eps must be in range [0, 1).")

        self.p_score = np.clip(self.learner.predict_proba(X_df)[:, 1], eps, 1 - eps)

        # 入力が1次元配列の場合は2次元に変換する
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.y_control = np.zeros(y.shape)
        self.y_treat = np.zeros(y.shape)

        # treatment=TrueとFalseの両方のサンプルが存在することを確認
        if np.sum(treatment) == 0 or np.sum(~treatment) == 0:
            raise ValueError("Both treatment and control groups must have samples.")

        # pandasのDataFrameからtreatment=TrueとFalse用のサブセットを抽出
        X_treat = X_df.iloc[np.where(treatment)[0]]
        X_control = X_df.iloc[np.where(~treatment)[0]]

        # Fit second models
        for i, _y in enumerate(y.T):
            y_treat = _y[treatment]
            y_control = _y[~treatment]

            self.treat_learner.fit(X_treat, y_treat)
            self.control_learner.fit(X_control, y_control)

            # 予測時はindexではなくマスクを使用
            self.y_control[:, i] = np.where(
                ~treatment, _y, self.control_learner.predict(X_df)
            )
            self.y_treat[:, i] = np.where(
                treatment, _y, self.treat_learner.predict(X_df)
            )

    def estimate_effect(
        self, treatment: np.ndarray, mode: str = "ate"
    ) -> tuple[float, float, float]:
        """
        Calculate the treatment effect using double robust method.

        Parameters
        ----------
        treatment : numpy.ndarray[bool]
            Flags with or without intervention.
        mode : str, default="ate"
            Adjustment method. Must be 'raw', 'ate', 'att' or 'atu'.

        Returns
        -------
        tuple
            A tuple containing (avg_y_control, avg_y_treat, effect_size)

        Raises
        ------
        ValueError
            If model is not fitted or mode is invalid.
        """
        if not hasattr(self, "y_control") or not hasattr(self, "y_treat"):
            raise ValueError("Model not fitted. Call fit() first.")

        self._check_mode(mode)
        weight = self.get_weight(treatment, mode=mode)
        return self._estimate_outcomes(weight)

    def _estimate_outcomes(self, weight: np.ndarray) -> tuple[float, float, float]:
        """
        Calculate outcome estimates using double robust method.

        Parameters
        ----------
        weight : numpy.ndarray
            Sample weight of IPW.

        Returns
        -------
        tuple
            A tuple containing (avg_y_control, avg_y_treat, effect_size)
        """
        # Use scalar averaging if possible to avoid array comparison issues
        avg_y_control = np.average(self.y_control, axis=0, weights=weight)
        avg_y_treat = np.average(self.y_treat, axis=0, weights=weight)
        effect_size = avg_y_treat - avg_y_control

        # Convert to scalar if single dimension to avoid array comparison issues
        if hasattr(effect_size, "__len__") and len(effect_size) == 1:
            avg_y_control = float(avg_y_control)
            avg_y_treat = float(avg_y_treat)
            effect_size = float(effect_size)

        return (avg_y_control, avg_y_treat, effect_size)
