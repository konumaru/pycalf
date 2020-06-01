import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
plt.style.use('seaborn')


class BaseModel:
    def __init__(self, learner):
        self.learner = learner
        self.ps = None
        self.weight = None

    def fit(self):
        raise NotImplementedError

    def raw_effect(self, treatment, outcomes):
        dummy_weight = np.ones(treatment.shape[0])

        effect_size = self._estimate_effect_size(treatment, outcomes, dummy_weight)
        effect_size = effect_size.assign(raw_effect=effect_size['Z1'] - effect_size['Z0'])
        return effect_size

    def acu(self, treatment):
        return roc_auc_score(treatment, self.ps)

    def plot_roc_curve(self, treatment, figsize=(7, 6)):
        fpr, tpr, thresholds = metrics.roc_curve(treatment, self.ps)
        auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_propensity_score(self, treatment, figsize=(12, 6)):
        plt.figure(figsize=figsize)
        plt.title('Propensity Score Distoribution.')
        plt.xlabel('Propensity Score')
        plt.ylabel('The Number of Data')
        plt.hist(
            self.ps[treatment == 0],
            bins=np.linspace(0, 1, 100, endpoint=False),
            rwidth=0.4,
            align='left',
            color='tab:blue'
        )
        plt.hist(
            self.ps[treatment == 1],
            bins=np.linspace(0, 1, 100, endpoint=False),
            rwidth=0.4,
            align='mid',
            color='tab:orange'
        )
        plt.show()


class IPW(BaseModel):

    def __init__(self, learner, clip_bounds=(1e-3, 1 - 1e-3)):
        self.clip_bounds = clip_bounds
        super().__init__(learner)

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
