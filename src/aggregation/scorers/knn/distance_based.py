from typing import Dict

import numpy as np

from src.aggregation.data.objects import Query
from src.aggregation.scorers.knn.base import KNN


class AdjustedAveragingMethod(KNN):
    def __init__(self, alpha: int = -1):
        # -1 <= alpha < n_c, 0 in article -> -1
        super(AdjustedAveragingMethod, self).__init__(None)
        self.alpha = alpha

    def score(self, query: Query) -> Dict[int, float]:
        issue_dists = self._get_issue_dists_dict(query)

        scores = dict()
        for issue, dists in issue_dists.items():
            dists.sort()

            if self.alpha + 1 > len(dists) - 1:
                scores[issue] = -float("inf")
                continue

            start = min(self.alpha + 1, len(dists) - 1)
            scores[issue] = -min(sum(dists[:i + 1]) / (i - self.alpha) for i in range(start, len(dists)))

        return scores


class AdjustedWeightingMethod(KNN):
    def __init__(self, omega: float, gamma: int):
        # omega <= 1
        # gamma <= min({n_c})
        super(AdjustedWeightingMethod, self).__init__(None)
        self.omega = omega
        self.gamma = gamma

    def score(self, query: Query) -> Dict[int, float]:
        issue_dists = self._get_issue_dists_dict(query)

        scores = dict()
        for issue, dists in issue_dists.items():
            dists.sort()
            max_range = min(self.gamma + 1, len(dists))
            scores[issue] = -sum(self.omega ** (i + 1) * dists[i] for i in range(max_range))

        return scores


class TruncatedPotentialsMethod(KNN):
    def __init__(self, beta: int):
        # beta <= min({n_c})
        super(TruncatedPotentialsMethod, self).__init__(None)
        self.beta = beta

    def score(self, query: Query) -> Dict[int, float]:
        issue_dists = self._get_issue_dists_dict(query)

        scores = dict()
        for issue, dists in issue_dists.items():
            dists.sort()
            max_range = min(self.beta + 1, len(dists))
            scores[issue] = sum(np.exp(-dists[i]) for i in range(max_range))

        return scores
