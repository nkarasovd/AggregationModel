from abc import abstractmethod, ABC
from functools import lru_cache
from typing import Any, Dict, List

from src.aggregation.data.objects import Query
from src.aggregation.data.scaler import Scaler
from src.aggregation.features.features_builder import FeaturesBuilder
from src.aggregation.features.features_config import FeaturesConfig


class AggregationScorer(ABC):
    @staticmethod
    @abstractmethod
    def load(config: Dict[str, Any]) -> 'AggregationScorer':
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path: str):
        raise NotImplementedError

    @abstractmethod
    def score(self, query: Query) -> Dict[int, float]:
        raise NotImplementedError


class FeatureScorer(AggregationScorer, ABC):
    def __init__(self, features_config: FeaturesConfig, scaler: Scaler):
        self.features_config = features_config
        self.features_builder = FeaturesBuilder(features_config)
        self.scaler = scaler

    @lru_cache(maxsize=25_000)
    def build_issues_features(self, query: Query) -> Dict[int, List[float]]:
        issues_features = self.features_builder.get_features(query)
        return self.scaler.transform(issues_features)
