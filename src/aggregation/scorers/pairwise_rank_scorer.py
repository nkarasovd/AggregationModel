import json
import os
from typing import Any, Dict, List

import torch
from torch import FloatTensor

from src.aggregation.data.objects import Query
from src.aggregation.data.scaler import Scaler
from src.aggregation.features.features_config import FeaturesConfig
from src.aggregation.scorers.rank_model import get_rank_model
from src.aggregation.scorers.rank_model.rank_model import RankModel
from src.aggregation.scorers.scorer import FeatureScorer


class PairwiseRankScorer(FeatureScorer):
    def __init__(self, rank_model: RankModel, features_config: FeaturesConfig,
                 scaler: Scaler, device: torch.device = None):
        super().__init__(features_config, scaler)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank_model = rank_model.to(self.device)

    @staticmethod
    def load(config: Dict[str, Any]) -> FeatureScorer:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank_model = get_rank_model(config['model'])
        features_config = FeaturesConfig.load_from_config(config['features'])
        scaler = Scaler.load_from_config(config['scaler'])

        return PairwiseRankScorer(rank_model, features_config, scaler, device)

    def save(self, model_path: str):
        config_path = os.path.join(model_path, "agg.json")

        data_dump = {
            'scorer_type': 'pairwise_scorer',
            'features': self.features_config.get_config(),
            'model': self.rank_model.get_config(),
            'scaler': self.scaler.get_config()
        }

        with open(config_path, 'w') as config_file:
            json.dump(data_dump, config_file)

    def score_by_features(self, issues_features: Dict[int, List[float]]) -> Dict[int, float]:
        scores, batch_size = {}, 128
        # {1: 2, 3: 4, 5: 6} -> [(1, 2), (3, 4), (5, 6)]
        list_key_value = list(issues_features.items())

        for i in range(0, len(list_key_value), batch_size):
            ids, features = zip(*list_key_value[i:i + batch_size])
            batch_scores = self.rank_model.predict(FloatTensor(features).to(self.device)).cpu().numpy()[:, 0]
            for issue_id, score in zip(ids, batch_scores):
                scores[issue_id] = score

        return scores

    def score(self, query: Query) -> Dict[int, float]:
        issues_features = self.build_issues_features(query)
        return self.score_by_features(issues_features)
