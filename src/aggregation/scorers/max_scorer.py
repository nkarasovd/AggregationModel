import json
import os
from typing import Any, Dict

from src.aggregation.data.objects import Query
from src.aggregation.scorers.scorer import AggregationScorer


class MaxScorer(AggregationScorer):
    @staticmethod
    def load(config: Dict[str, Any]) -> AggregationScorer:
        return MaxScorer()

    def save(self, model_path: str):
        config_path = os.path.join(model_path, "agg.json")

        data_dump = {
            'scorer_type': 'max_scorer'
        }

        with open(config_path, 'w') as config_file:
            json.dump(data_dump, config_file)

    def score(self, query: Query) -> Dict[int, float]:
        return {
            issue_dists.issue_id: max(issue_dists.dists, key=lambda x: x.dist).dist
            for issue_dists in query.issues
        }
