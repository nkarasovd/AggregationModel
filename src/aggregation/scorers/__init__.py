import json
import os

from src.aggregation.scorers.linear_scorer import LinearScorer
from src.aggregation.scorers.max_scorer import MaxScorer
from src.aggregation.scorers.pairwise_rank_scorer import PairwiseRankScorer
from src.aggregation.scorers.scorer import AggregationScorer


def load_scorer(model_path: str) -> AggregationScorer:
    config_path = os.path.join(model_path, "agg.json")

    with open(config_path) as config_file:
        config = json.load(config_file)

        if config['scorer_type'] == 'max_scorer':
            return MaxScorer.load(config)
        elif config['scorer_type'] == 'linear_scorer':
            return LinearScorer.load(config)
        elif config['scorer_type'] == 'pairwise_scorer':
            return PairwiseRankScorer.load(config)
        else:
            raise ValueError('Scorer name is not match')
