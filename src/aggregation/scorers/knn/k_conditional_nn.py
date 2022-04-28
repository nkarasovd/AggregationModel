from typing import Dict, Optional

from src.aggregation.data.objects import Query
from src.aggregation.scorers.knn.base import KNN


class KConditionalNN(KNN):
    def __init__(self, k: Optional[int], q: int = 1):
        super(KConditionalNN, self).__init__(k)
        self.q = q

    def score(self, query: Query) -> Dict[int, float]:
        issue_dists = self._get_issue_dists_dict(query, shift=1e-6)

        issue_kth_dist = {}
        for issue, dists in issue_dists.items():
            dists.sort()
            issue_kth_dist[issue] = dists[min(self.k, len(dists)) - 1] ** (-self.q)

        denominator = sum(kth_dist for kth_dist in issue_kth_dist.values())

        scores = {issue: kth_dist / denominator for issue, kth_dist in issue_kth_dist.items()}

        return scores
