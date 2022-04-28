from typing import List


class FeaturesConfig:
    features_dim = {'hist_10_bins_unique': 10,
                    'hist_10_bins_unique_weighted': 10,
                    'hist_12_bins': 12,
                    'hist_12_bins_weighted': 12,
                    'hist_10_bins_weights': 10
                    }

    def __init__(self, features: List[str]):
        self.features = features

    @staticmethod
    def build_features_config(features_str: str) -> 'FeaturesConfig':
        if features_str == 'first_max':
            return first_max
        elif features_str == 'similarity_features':
            return similarity_features
        elif features_str == 'best_features':
            return best_features
        else:
            raise ValueError("FeaturesConfig name is not match")

    @staticmethod
    def load_from_config(config: List[str]) -> 'FeaturesConfig':
        return FeaturesConfig(config)

    def get_config(self) -> List[str]:
        return self.features

    def get_feature_dim(self, feature: str) -> int:
        return self.features_dim[feature] if feature in self.features_dim else 1

    def get_config_dim(self) -> int:
        return sum(self.get_feature_dim(feature) for feature in self.features)


first_max = FeaturesConfig(['first_max'])
similarity_features = FeaturesConfig(['first_max', 'hist_12_bins'])
best_features = FeaturesConfig(['first_max', 'hist_12_bins_weighted', 'max_weight',
                                'min_weight', 'mean_weight', 'hist_10_bins_weights',
                                'difference_max_min', 'ts_on_max_score'])
