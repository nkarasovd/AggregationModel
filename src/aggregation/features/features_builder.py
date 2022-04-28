from typing import List, Tuple, Union

import numpy as np

from src.aggregation.data.objects import DistInfo, Query
from src.aggregation.features.features_config import FeaturesConfig


class FeaturesBuilder:
    """
    Use FeaturesBuilder class to calculate features for Aggregation model
    """

    def __init__(self, features_config: FeaturesConfig):
        self.features_map = {'first_max': self._first_max,
                             'second_max': self._second_max,
                             'mean': self._mean,
                             'mean_unique': self._mean_unique,
                             'mean_90_100': self._mean_90_100,
                             'mean_unique_90_100': self._mean_unique_90_100,
                             'reports_num': self._reports_num,
                             'unique_reports_num': self._unique_reports_num,
                             'std': self._std,
                             'std_unique': self._std_unique,
                             'quantile_25': self._quantile_25,
                             'quantile_75': self._quantile_75,
                             'quantile_25_unique': self._quantile_25_unique,
                             'quantile_75_unique': self._quantile_75_unique,
                             'hist_10_bins_unique': self._hist_10_bins_unique,
                             'hist_10_bins_unique_weighted': self._hist_10_bins_unique_weighted,
                             'hist_10_bins_weights': self._hist_10_bins_weights,
                             'mean_weight_unique': self._mean_weight_unique,
                             'mean_weight': self._mean_weight,
                             'ts_on_max_score': self._ts_on_max_score,
                             'hist_12_bins': self._hist_12_bins,
                             'hist_12_bins_weighted': self._hist_12_bins_weighted,
                             'max_weight': self._max_weight,
                             'min_weight': self._min_weight,
                             'difference_max_min': self._difference_max_min
                             }
        self.features = features_config.features
        self.scores = None
        self.unique_scores = None
        self.weights = None
        self.weights_unique = None
        self.max_score_ts = None

    def _first_max(self) -> float:
        # return self.unique_scores[-1]
        return self.max_score

    def _second_max(self) -> float:
        return self.unique_scores[-2 if len(self.unique_scores) > 1 else -1]

    def _mean(self) -> np.ndarray:
        return np.mean(self.scores)

    def _mean_unique(self) -> np.ndarray:
        return np.mean(self.unique_scores)

    def _mean_90_100(self) -> np.ndarray:
        return np.mean(self.scores[int(len(self.scores) * 0.9):])

    def _mean_unique_90_100(self) -> np.ndarray:
        return np.mean(self.unique_scores[int(len(self.unique_scores) * 0.9):])

    def _reports_num(self) -> int:
        return len(self.scores)

    def _unique_reports_num(self) -> int:
        return len(self.unique_scores)

    def _std(self) -> np.ndarray:
        return np.std(self.scores)

    def _std_unique(self) -> np.ndarray:
        return np.std(self.unique_scores)

    def _quantile_25(self) -> float:
        return np.quantile(self.scores, 0.25)

    def _quantile_75(self) -> float:
        return np.quantile(self.scores, 0.75)

    def _quantile_25_unique(self) -> float:
        return np.quantile(self.unique_scores, 0.25)

    def _quantile_75_unique(self) -> float:
        return np.quantile(self.unique_scores, 0.75)

    def _hist_12_bins(self) -> Union[List[int], List[float]]:
        hist = np.histogram(self.scores, bins=12)[0]
        return [h for h in hist]

    def _hist_12_bins_weighted(self) -> Union[List[int], List[float]]:
        hist = np.histogram(self.scores, bins=12, weights=self.weights)[0]
        return [h for h in hist]

    def _hist_10_bins_unique(self) -> Union[List[int], List[float]]:
        hist = np.histogram(self.unique_scores, bins=10)[0]
        return [h for h in hist]

    def _hist_10_bins_unique_weighted(self) -> Union[List[int], List[float]]:
        hist = np.histogram(self.unique_scores, bins=10, weights=self.weights_unique)[0]
        return [h for h in hist]

    def _hist_10_bins_weights(self) -> List[int]:
        hist = np.histogram(self.weights, bins=10)[0]
        return [h for h in hist]

    def _mean_weight(self) -> np.ndarray:
        return np.mean(self.weights)

    def _mean_weight_unique(self) -> np.ndarray:
        return np.mean(self.weights_unique)

    def _max_weight(self) -> float:
        return max(self.weights)

    def _ts_on_max_score(self) -> float:
        return 1.0 / (np.log(self.max_score_ts + 1) + 1)

    def _min_weight(self) -> float:
        return min(self.weights)

    def _difference_max_min(self) -> float:
        return (self._max_weight() - self._min_weight()) / self._max_weight()

    def _get_point(self, dist_infos: Tuple[DistInfo]) -> List[float]:
        result = []

        self._build_features(dist_infos)

        for feature in self.features:
            value = self.features_map[feature]()
            result.extend(value if isinstance(value, list) else [value])

        return result

    def _build_features(self, dist_infos: Tuple[DistInfo]):
        scores, scores_unique = [], []

        timestamps, timestamps_unique = [], []

        for dist_info in dist_infos:
            scores += [float(dist_info.dist)] * dist_info.count
            scores_unique += [float(dist_info.dist)]

            timestamps += [dist_info.ts_diff] * dist_info.count
            timestamps_unique += [dist_info.ts_diff]

        weights = 1.0 / (np.log(np.array(timestamps) + 1) + 1)
        weights_unique = 1.0 / (np.log(np.array(timestamps_unique) + 1) + 1)

        argmax = np.argmax(scores_unique)
        self.max_score_ts = timestamps_unique[argmax]
        self.max_score = scores_unique[argmax]

        scores_ids = np.argsort(scores)
        self.scores = np.array(scores)[scores_ids]
        self.weights = np.array(weights)[scores_ids]

        scores_unique_ids = np.argsort(scores_unique)
        self.unique_scores = np.array(scores_unique)[scores_unique_ids]
        self.weights_unique = np.array(weights_unique)[scores_unique_ids]

    def get_features(self, query: Query):
        return {
            issue_dist.issue_id: self._get_point(issue_dist.dists)
            for issue_dist in query.issues
        }
