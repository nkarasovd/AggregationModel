from abc import abstractmethod, ABC
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor


class RankModel(nn.Module, ABC):
    def __init__(self, model: nn.Module, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1: Tensor, input_2: Tensor) -> Tensor:
        s1 = self.model(input_1)
        s2 = self.model(input_2)
        out = self.sigmoid(s2 - s1)
        return torch.cat((1 - out, out), dim=1).view(-1, 2)

    def predict(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model(x)

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_from_config(model_config: Dict[str, Any]) -> 'RankModel':
        raise NotImplementedError

    @abstractmethod
    def print_info(self):
        raise NotImplementedError
