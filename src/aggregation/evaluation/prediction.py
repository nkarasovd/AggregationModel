from typing import Dict, Iterable, List, Tuple

from src.aggregation.data.objects import LabeledQuery
from src.aggregation.scorers.scorer import AggregationScorer


def predict(model: AggregationScorer,
            labeled_queries: List[LabeledQuery]) -> Iterable[Tuple[int, Dict[int, float]]]:
    """
    Return right issue_id and dict of scores
    @param model: Aggregation model
    @param labeled_queries: List of LabeledQuery
    """
    for labeled_query in labeled_queries:
        yield labeled_query.issue_id, model.score(labeled_query.query)
