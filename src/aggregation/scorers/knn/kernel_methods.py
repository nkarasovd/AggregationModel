from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np

from src.aggregation.data.objects import Query
from src.aggregation.scorers.knn.base import KNN


class KernelKNN(KNN, ABC):
    def __init__(self, k: Optional[int] = None):
        super(KernelKNN, self).__init__(k)

    @staticmethod
    @abstractmethod
    def _kernel(dist: float) -> float:
        raise NotImplementedError

    def score(self, query: Query) -> Dict[int, float]:
        distances = self._get_distances(query)

        scores = {issue: 0 for issue, _ in distances}
        for issue_id, dist in distances:
            scores[issue_id] = scores.get(issue_id, 0) + self._kernel(dist)

        return scores


class UniformKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 1 if dist < 1 else 0


class TriangleKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 1 - dist if dist < 1 else 0


class EpanechnikovKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 0.75 * (1 - dist ** 2) if dist < 1 else 0


class QuarticKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 0.9375 * (1 - dist ** 2) ** 2 if dist < 1 else 0


class TriweightKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 1.09375 * (1 - dist ** 2) ** 3 if dist < 1 else 0


class GaussianKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return np.exp(-0.5 * dist ** 2) / np.sqrt(2 * np.pi)


class CosineKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return np.pi / 4 * np.cos(np.pi / 2 * dist) if dist < 1 else 0


class TricubeKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 70.0 / 81.0 * (1 - dist ** 3) ** 3 if dist < 1 else 0


class LogisticKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 1.0 / (np.exp(dist) + 2 + np.exp(-dist))


class SigmoidKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 2.0 / np.pi * (1.0 / (np.exp(dist) + np.exp(-dist)))


class SilvermanKernelKNN(KernelKNN):
    @staticmethod
    def _kernel(dist: float) -> float:
        return 0.5 * np.exp(-dist / np.sqrt(2)) * np.sin(dist / np.sqrt(2) + np.pi / 4.0)
