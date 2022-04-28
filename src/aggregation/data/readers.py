import json
from typing import List

from tqdm import tqdm

from src.aggregation.data.objects import DistInfo, IssueDists, Query, LabeledQuery


def read_data(filename: str, size: int = 25000, train_mode: bool = False) -> List[LabeledQuery]:
    res = []
    with open(filename) as f:
        for i, line in enumerate(tqdm(f)):
            if i >= size:
                break
            event = json.loads(line)

            issues_scores, issues = dict(), []
            for k, v in event["issues"].items():
                # keys in json are strings
                issues_scores[int(k)] = v
                issue_dists = tuple(DistInfo(float(dist), hist[0], hist[1]) for dist, hist in v.items())
                issues.append(IssueDists(int(k), issue_dists))

            query = LabeledQuery(event["right"], Query(tuple(issues)))
            if train_mode:
                if event["right"] in issues_scores and len(issues_scores[event["right"]]) != 0:
                    res.append(query)
            else:
                res.append(query)
    return res
