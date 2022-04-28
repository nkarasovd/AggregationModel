from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

from hyperopt import hp, fmin, tpe, space_eval

from src.similarity.data.buckets.bucket_data import BucketData, DataSegment
from src.similarity.evaluation.issue_sim import map_metric
from src.similarity.evaluation.stack_sim import auc_model
from src.similarity.methods.base import SimStackModel
from src.similarity.methods.pair_stack_issue_model import PairStackBasedSimModel


class HyperoptModel(ABC):
    @abstractmethod
    def set_params(self, args: Dict[str, float]):
        raise NotImplementedError

    @abstractmethod
    def score_model(self, data: BucketData, val: DataSegment):
        raise NotImplementedError

    @abstractmethod
    def params_edges(self) -> Dict[str, Tuple[float, float]]:
        raise NotImplementedError

    def find_hyperparams(self, data: BucketData, val: DataSegment, max_evals: int = 20):
        def objective(args):
            self.set_params(args)
            return self.score_model(data, val)

        params = self.params_edges()
        if params is None or len(params) == 0:
            return
        space = {name: hp.uniform(name, edges[0], edges[1]) for name, edges in self.params_edges().items()}
        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)
        print("Top params", space_eval(space, best))
        self.set_params(space_eval(space, best))


class SimStackHyperoptModel(SimStackModel, HyperoptModel, ABC):
    def set_params(self, args: Dict[str, float]):
        for k, v in args.items():
            self.__dict__[k] = v

    def score_model(self, data: BucketData, val: DataSegment):
        return 1 - auc_model(self, data, val, full=False)[0]

    def find_params(self, sim_val_data: List[Tuple[int, int, int]]) -> 'SimStackHyperoptModel':
        self.find_hyperparams(sim_val_data)
        return self


class PairStackBasedIssueHyperoptModel(PairStackBasedSimModel, HyperoptModel):
    def params_edges(self) -> Dict[str, Tuple[float, float]]:
        try:
            return self.stack_model.params_edges()
        except:
            return {}

    def set_params(self, args: Dict[str, float]):
        for k, v in args.items():
            self.stack_model.__dict__[k] = v

    def score_model(self, data: BucketData, val: DataSegment):
        new_preds = self.predict(data.get_events(val))
        score = map_metric(new_preds)
        print("MRR:", score)
        return 1 - score

    def find_params(self, data: BucketData, val: DataSegment) -> 'PairStackBasedIssueHyperoptModel':
        self.find_hyperparams(data, val)
        return self
