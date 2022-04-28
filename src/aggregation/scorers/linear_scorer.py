import json
import os
from typing import Any, Dict

import numpy as np

from src.aggregation.data.objects import Query
from src.aggregation.data.scaler import Scaler
from src.aggregation.features.features_config import FeaturesConfig
from src.aggregation.scorers.scorer import FeatureScorer


class LinearScorer(FeatureScorer):
    def __init__(self, features_config: FeaturesConfig, weights: np.ndarray,
                 scaler: Scaler, bias: float = None):
        super().__init__(features_config, scaler)
        self.weights = weights
        self.bias = bias

    @staticmethod
    def load(config: Dict[str, Any]) -> FeatureScorer:
        features_config = FeaturesConfig.load_from_config(config['features'])
        weights = np.array(config['model']['weights']).reshape(-1, 1)
        scaler = Scaler.load_from_config(config['scaler'])
        bias = config['model']['bias']

        return LinearScorer(features_config, weights, scaler, bias)

    def save(self, model_path: str):
        config_path = os.path.join(model_path, "agg.json")

        model_config = {
            'model_type': 'linear_model',
            'weights': self.weights.tolist(),
            'bias': self.bias
        }

        data_dump = {
            'scorer_type': 'linear_scorer',
            'features': self.features_config.get_config(),
            'model': model_config,
            'scaler': self.scaler.get_config()
        }

        with open(config_path, 'w') as config_file:
            json.dump(data_dump, config_file)

    def score(self, query: Query) -> Dict[int, float]:
        issues_features = self.build_issues_features(query)
        return {
            issue_id: (np.array(feature_vector).reshape(1, -1) @ self.weights).item()
            for issue_id, feature_vector in issues_features.items()
        }
