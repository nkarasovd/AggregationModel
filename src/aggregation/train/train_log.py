from typing import Dict, Tuple

from src.aggregation.scorers.rank_model.rank_model import RankModel


class BestResult:
    color_start = '\033[92m'
    color_end = '\033[0m'

    def __init__(self, epoch: int, rank_model: RankModel,
                 scores: Dict[str, Tuple[float, float, float]]):
        self.epoch = epoch
        self.scores = scores
        self.rank_model = rank_model

    def print_log(self):
        print('\n\nBest result:')
        print(f'{self.color_start}Epoch = {self.epoch + 1}{self.color_end}')

        for name, score in self.scores.items():
            print(f"{self.color_start}{name}: {score}")

        self.rank_model.print_info()
        print(f'{self.color_end}', end='')
