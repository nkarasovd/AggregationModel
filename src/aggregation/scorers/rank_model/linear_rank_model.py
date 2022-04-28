from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from src.aggregation.scorers.rank_model.rank_model import RankModel


class LinearRankModel(RankModel):
    def __init__(self, input_dim: int, output_dim: int = 1):
        model = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        super().__init__(model, input_dim, output_dim)

    def get_config(self) -> Dict[str, Any]:
        model_config = {
            'model_type': 'linear_model',
            'weights': self.model[0].weight[0].detach().cpu().tolist(),
            'bias': self.model[0].bias[0].item()
        }

        return model_config

    @staticmethod
    def load_from_config(model_config: Dict[str, Any]) -> RankModel:
        weights = np.array(model_config['weights']).reshape(-1, 1)
        bias = model_config['bias']

        rank_model = LinearRankModel(weights.shape[0])

        with torch.no_grad():
            x = torch.from_numpy(weights).float()
            rank_model.model[0].weight.data = x
            rank_model.model[0].bias.data = torch.from_numpy(np.array([bias])).float()

        return rank_model

    def print_info(self):
        step = 7
        weights = self.model[0].weight.data[0].tolist()

        print('\nFeatures weights:')

        for i in range(0, len(weights), step):
            print(list(map(lambda x: round(x, 4), weights[i: i + step])))
        print()
