from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from src.similarity.data.objects import Issue
from src.similarity.methods.base import SimStackModel, IssueScorer


class HypothesisSelector(ABC):
    @abstractmethod
    def filter_top(self, event_id: int, st_id: int, issues: Dict[int, Issue]) -> Dict[int, List[int]]:
        raise NotImplementedError


class ScoresBasedHypothesisSelector(HypothesisSelector, ABC):
    def __init__(self, stack_model: SimStackModel, issue_scorer: IssueScorer, *args, **kwargs):
        self.stack_model = stack_model
        self.issue_scorer = issue_scorer

    def score_stacks(self, event_id: int, st_id: int, issues: Dict[int, Issue]) -> List[Tuple[int, int, float]]:
        res = []
        for id, issue in issues.items():
            stacks = issue.confident_state()
            if len(stacks) == 0:
                continue
            stack_ids = [st.id for st in stacks]
            preds = self.stack_model.predict(st_id, stack_ids)
            res += list(zip([id] * len(stack_ids), stack_ids, preds))
        res = sorted(res, key=lambda x: -x[2])
        return res


class CachedHypothesisSelector(HypothesisSelector, ABC):
    def __init__(self, *args, **kwargs):
        self.cache = {}

    @abstractmethod
    def _filter_top(self, event_id: int, st_id: int, issues: Dict[int, Issue]) -> Dict[int, List[int]]:
        raise NotImplementedError

    def filter_top(self, event_id: int, st_id: int, issues: Dict[int, Issue]) -> Dict[int, List[int]]:
        cache_key = event_id, len(issues)
        if cache_key not in self.cache:
            self.cache[cache_key] = self._filter_top(event_id, st_id, issues)
        return self.cache[cache_key]


class TopNumHypothesisSelector(CachedHypothesisSelector, ScoresBasedHypothesisSelector):
    """
    Take at least top_issues issues and at least top_stacks in order of descending similarity
    """
    def __init__(self, stack_model: SimStackModel, issue_scorer: IssueScorer, top_stacks: int = None,
                 top_issues: int = None):
        ScoresBasedHypothesisSelector.__init__(self, stack_model, issue_scorer)
        CachedHypothesisSelector.__init__(self)
        assert top_stacks is not None and top_issues is not None
        self.top_stacks = top_stacks
        self.top_issues = top_issues

    def __str__(self) -> str:
        return f"{self.stack_model.name()}_{self.issue_scorer.name()}_ts{self.top_stacks}_ti{self.top_issues}"

    def _filter_top(self, event_id: int, st_id: int, issues: Dict[int, Issue]) -> Dict[int, List[int]]:
        stack_scores = self.score_stacks(event_id, st_id, issues)
        filtered_issues = {}
        stacks_cnt = 0
        for id, st_id, pred in stack_scores:
            if id not in filtered_issues:
                filtered_issues[id] = [st_id]
                stacks_cnt += 1
            elif stacks_cnt < self.top_stacks:
                filtered_issues[id].append(st_id)
                stacks_cnt += 1
            if stacks_cnt >= self.top_stacks and len(filtered_issues) >= self.top_issues:
                break
        return filtered_issues


class TopIssuesHypothesisSelector(ScoresBasedHypothesisSelector, CachedHypothesisSelector):
    def __init__(self, stack_model: SimStackModel, issue_scorer: IssueScorer, top_issues: int = None):
        ScoresBasedHypothesisSelector.__init__(self, stack_model, issue_scorer)
        CachedHypothesisSelector.__init__(self)
        assert top_issues is not None
        self.top_issues = top_issues

    def __str__(self) -> str:
        return f"{self.stack_model.name()}_{self.issue_scorer.name()}_ti{self.top_issues}"

    def _filter_top(self, event_id: int, st_id: int, issues: Dict[int, Issue]) -> Dict[int, List[int]]:
        stack_scores = self.score_stacks(event_id, st_id, issues)
        filtered_issues = {}
        for id, st_id, pred in stack_scores:
            if id not in filtered_issues:
                filtered_issues[id] = []
            if len(filtered_issues) >= self.top_issues:
                break
        for is_id in filtered_issues:
            filtered_issues[is_id] = [st.id for st in issues[is_id].confident_state()]
        return filtered_issues


class ScoreThrHypothesisSelector(ScoresBasedHypothesisSelector, CachedHypothesisSelector):
    def __init__(self, stack_model: SimStackModel, issue_scorer: IssueScorer, thr: float):
        ScoresBasedHypothesisSelector.__init__(self, stack_model, issue_scorer)
        CachedHypothesisSelector.__init__(self)
        self.thr = thr

    def _filter_top(self, event_id: int, st_id: int, issues: Dict[int, Issue]) -> Dict[int, List[int]]:
        stack_scores = self.score_stacks(event_id, st_id, issues)
        filtered_issues = {}
        for id, st_id, pred in stack_scores:
            if pred > self.thr:
                if id not in filtered_issues:
                    filtered_issues[id] = []
                filtered_issues[id].append(st_id)
        return filtered_issues
