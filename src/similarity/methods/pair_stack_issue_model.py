from typing import Dict, Tuple, Iterable, Union

from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.data.objects import Issue
from src.similarity.methods.base import SimIssueModel, SimStackModel, IssueScorer
from src.similarity.methods.hypothesis_selection import HypothesisSelector


class MaxIssueScorer(IssueScorer):
    def score(self, scores: Iterable[float], with_arg: bool = False) -> Union[float, Tuple[float, int]]:
        if with_arg:
            ind = 0
            value = None
            for i, score in enumerate(scores):
                if value is None or score > value:
                    ind = i
                    value = score
            return value, ind
        return max(scores)

    def name(self) -> str:
        return "max_scorer"


class PairStackBasedSimModel(SimIssueModel):
    def __init__(self, stack_model: SimStackModel, issue_scorer: IssueScorer, filter_model: HypothesisSelector = None):
        self.stack_model = stack_model
        self.issue_scorer = issue_scorer
        self.filter_model = filter_model

    def name(self) -> str:
        return "_".join([model.name() for model in [self.stack_model, self.issue_scorer, self.filter_model]])

    def predict_all(self, st_id: int, issues: Dict[int, Issue]) -> Tuple[Dict[int, Union[float, Tuple[float, int]]], int]:
        pred_issues = {}
        stacks_cnt = 0
        for id, issue in issues.items():
            stacks = issue.confident_state()
            stacks_cnt += len(stacks)
            if len(stacks) == 0:
                pred_issues[id] = 0
                continue
            preds = self.stack_model.predict(st_id, [st.id for st in stacks])
            score = self.issue_scorer.score(preds, with_arg=False)
            pred_issues[id] = score
        return pred_issues, stacks_cnt

    def predict_filtered(self, event_id: int, st_id: int, issues: Dict[int, Issue], with_stacks: bool = False) \
            -> Tuple[Dict[int, Union[float, Tuple[float, int]]], int]:
        pred_issues = {}
        stacks_cnt = 0
        top_issues = self.filter_model.filter_top(event_id, st_id, issues)
        min_score = float("inf")
        for id, stack_ids in top_issues.items():
            preds = self.stack_model.predict(st_id, stack_ids)
            score = self.issue_scorer.score(preds, with_arg=with_stacks)
            if with_stacks:
                pred_issues[id] = score[0], stack_ids[score[1]]
                min_score = min(min_score, score[0])
            else:
                pred_issues[id] = score
                min_score = min(min_score, score)
            stacks_cnt += len(stack_ids)

        # save other issues for fair map and other metrics comparison
        min_score -= 1
        for is_id, issue in issues.items():
            if is_id not in top_issues:
                if with_stacks:
                    pred_issues[is_id] = min_score, list(issue.stacks.keys())[0]
                else:
                    pred_issues[is_id] = min_score
        return pred_issues, stacks_cnt

    def predict(self, events: Iterable[StackAdditionState]) -> Iterable[Tuple[int, Dict[int, float]]]:
        for i, event in enumerate(events):
            if self.filter_model:
                pred_issues, _ = self.predict_filtered(event.id, event.st_id, event.issues)
            else:
                pred_issues, _ = self.predict_all(event.st_id, event.issues)
            yield event.is_id, pred_issues
