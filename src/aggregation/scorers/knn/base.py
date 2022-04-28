from abc import ABC
from typing import List, Optional, Tuple, Dict, Any

from src.aggregation.data.objects import Query
from src.aggregation.scorers import AggregationScorer


class KNN(AggregationScorer, ABC):
    def __init__(self, k: Optional[int]):
        self.k = k

    @staticmethod
    def load(config: Dict[str, Any]) -> 'AggregationScorer':
        pass

    def save(self, model_path: str):
        pass

    @staticmethod
    def _get_max_similarity(query: Query) -> float:
        return max(dist_info.dist for issue_dists in query.issues for dist_info in issue_dists.dists)

    def _get_border(self, query: Query, max_similarity: float):
        assert self.k is not None, "k is None!"

        similarities = []
        for issue_dists in query.issues:
            similarities.extend([x.dist for x in issue_dists.dists])
        unique_similarities = sorted(set(similarities))

        if len(unique_similarities) >= self.k + 1:
            return max_similarity - unique_similarities[-(self.k + 1)]

        return max_similarity - unique_similarities[0] + 1

    @staticmethod
    def _scale(similarity: float, max_similarity: float,
               border: Optional[float] = None) -> float:
        if border is None:
            return max_similarity - similarity

        return (max_similarity - similarity) / border

    @staticmethod
    def _to_dist(similarity: float, max_similarity: float) -> float:
        return max_similarity - similarity

    def _get_distances(self, query: Query) -> List[Tuple[int, float]]:
        max_sim = self._get_max_similarity(query)

        distances = []
        for issue_dists in query.issues:
            issue_id = issue_dists.issue_id
            distances.extend([(issue_id, self._to_dist(x.dist, max_sim)) for x in issue_dists.dists])

        if self.k is not None:
            border = self._get_border(query, max_sim)
            distances = [(issue_id, dist / border) for issue_id, dist in distances]

        distances.sort(key=lambda x: x[1])

        return distances

    def _get_issue_dists_dict(self, query: Query, shift: float = 0) -> Dict[int, List[float]]:
        distances = self._get_distances(query)

        issue_dists = dict()
        for issue, dist in distances:
            issue_dists[issue] = issue_dists.get(issue, []) + [dist + shift]

        return issue_dists
