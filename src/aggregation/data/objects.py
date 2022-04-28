from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(eq=True, frozen=True)
class DistInfo:
    dist: float
    count: int
    ts_diff: Optional[int]


@dataclass(eq=True, frozen=True)
class IssueDists:
    issue_id: int
    dists: Tuple[DistInfo]

    def __len__(self):
        return len(self.dists)


@dataclass(eq=True, frozen=True)
class Query:
    issues: Tuple[IssueDists]


class LabeledQuery:
    def __init__(self, right_issue_id: int, query: Query):
        self.issue_id = right_issue_id
        self.query = query


@dataclass(eq=True, frozen=True)
class LabeledQueryFeatures:
    right_issue_id: int
    issues_features: Dict[int, List[float]]
