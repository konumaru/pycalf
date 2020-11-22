
import numpy as np


class UpliftModel():
    def __init__(self, learner_treat, learner_control):
        self.learner_treat = learner_treat
        self.learner_control = learner_control

    def fit(self, X_treat, y_treat, X_control, y_control, weight_treat=None, weight_control=None):
        self.learner_treat.fit(X_treat, y_treat, sample_weight=weight_treat)
        self.learner_control.fit(X_control, y_control, sample_weight=weight_control)

    def estimate_uplift_score(self, X):
        proba_treat = self.learner_treat.predict_proba(X)[:, 1]
        proba_control = self.learner_control.predict_proba(X)[:, 1]
        uplift_score = proba_treat / proba_control
        return uplift_score

    def predict(self, X, treatment, y):
        uplift_score = self.estimate_uplift_score(X)

        sorted_idx = np.argsort(uplift_score)[::-1]
        uplift_score = uplift_score[sorted_idx]
        y = y[sorted_idx]
        treatment = treatment[sorted_idx]

        y_treat = np.nancumsum(np.where(treatment == 1, y, np.nan))
        y_control = np.nancumsum(np.where(treatment == 0, y, np.nan))

        treat_size = np.nancumsum(np.where(treatment == 1, treatment, np.nan))
        control_size = np.nancumsum(np.where(treatment == 0, (1 - treatment), np.nan))

        cumavg_y_treat = np.array([
            0.0 if s == 0 else _y / s for _y, s in zip(y_treat, treat_size)])
        cumavg_y_control = np.array([
            0.0 if s == 0 else _y / s for _y, s in zip(y_control, control_size)])

        lift = (cumavg_y_treat - cumavg_y_control) * treat_size
        return (uplift_score, lift)

    def get_baseline(self, lift):
        data_size = len(lift)
        base_line = np.arange(data_size) * lift[data_size - 1] / data_size
        return base_line

    def get_auuc(self, lift):
        base_line = self.get_baseline(lift)
        auuc = (lift - base_line).sum() / len(lift)
        return auuc
