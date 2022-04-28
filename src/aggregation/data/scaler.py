from typing import Dict, List, Optional

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.aggregation.data.objects import Query
from src.aggregation.features.features_builder import FeaturesBuilder
from src.aggregation.features.features_config import FeaturesConfig


class Scaler:
    def __init__(self, scale: bool = False):
        self.scaler = StandardScaler() if scale else None

    def fit(self, train_queries: List[Query], features_config: FeaturesConfig) -> 'Scaler':
        if self.scaler is None:
            return self

        features_builder = FeaturesBuilder(features_config)
        features_list = []
        for query in tqdm(train_queries, position=0, leave=True):
            features_list += list(features_builder.get_features(query).values())

        self.scaler.fit(features_list)

        return self

    @staticmethod
    def load_from_config(scaler_config: Optional[Dict[str, float]]) -> 'Scaler':
        if scaler_config is not None:
            scaler = Scaler(True)
            scaler.scaler.mean_ = scaler_config['mean']
            scaler.scaler.var_ = scaler_config['var']
            return scaler

        return Scaler()

    def get_config(self) -> Optional[Dict[str, float]]:
        if self.scaler is not None:
            return {
                'mean': self.scaler.mean_,
                'var': self.scaler.var_
            }

    def transform(self, issues_features: Dict[int, List[float]]) -> Dict[int, List[float]]:
        if self.scaler is not None:
            return {k: self.scaler.transform([v])[0] for k, v in issues_features.items()}
        return issues_features
