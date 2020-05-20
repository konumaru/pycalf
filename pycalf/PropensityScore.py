import numpy as np
import pandas as pd


class IPW:
    def __init__(self, learner, clip_bounds=(1e-3, 1 - 1e-3)):
        self.learner = learner
        self.clip_bounds = clip_bounds
        self.ps = None
        self.weight = None

    def fit(self, X, treatment):
        self.learner.fit(X, treatment)
        self.ps = np.clip(self.learner.predict_proba(X)[:, 1], *self.clip_bounds)
        self.weight = np.where(treatment == 1, 1 / self.ps, 1 / (1 - self.ps))

    def _estimate_effect_size(self, treatment, outcomes, weight):
        treatment = np.array(treatment, dtype='bool')
        effect_size = pd.DataFrame(columns=['Z0', 'Z1'])

        for i, (name, values) in enumerate(outcomes.items()):
            effect_size.loc[name, 'Z0'] = np.average(values[~treatment], weights=weight[~treatment])
            effect_size.loc[name, 'Z1'] = np.average(values[treatment], weights=weight[treatment])

        return effect_size

    def estimate_ate(self, treatment, outcomes):
        ate_weight = self.weight

        effect_size = self._estimate_effect_size(treatment, outcomes, ate_weight)
        effect_size = effect_size.assign(ATE=effect_size['Z1'] - effect_size['Z0'])
        return effect_size

    def estimate_att(self, treatment, outcomes):
        att_weight = self.ps * self.weight

        effect_size = self._estimate_effect_size(treatment, outcomes, att_weight)
        effect_size = effect_size.assign(ATT=effect_size['Z1'] - effect_size['Z0'])
        return effect_size

    def estimate_atu(self, treatment, outcomes):
        atu_weight = (1 - self.ps) * self.weight

        effect_size = self._estimate_effect_size(treatment, outcomes, atu_weight)
        effect_size = effect_size.assign(ATU=effect_size['Z1'] - effect_size['Z0'])
        return effect_size
