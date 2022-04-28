import json
from typing import Iterable, Dict, List

import numpy as np
from tqdm import tqdm

from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.methods.base import SimStackModel
from src.similarity.methods.hypothesis_selection import HypothesisSelector


class DumpBuilder:
    def __init__(self, filter_model: HypothesisSelector, stack_model: SimStackModel):
        self.filter_model = filter_model
        self.stack_model = stack_model

    def dump_issue_scores_filtered(self, events: Iterable[StackAdditionState],
                                   filename: str, st_timestamps: Dict[int, int],
                                   build_features: bool = False):
        with open(filename, 'w') as f:
            for i, event in tqdm(enumerate(events)):
                event_id, st_id, issues = event.id, event.st_id, event.issues
                ts = st_timestamps[st_id]
                pred_issues = {}

                if self.filter_model:
                    top_issues = self.filter_model.filter_top(event_id, st_id, issues)
                    for id_, stack_ids in top_issues.items():
                        predictions = self.stack_model.predict(st_id, stack_ids)
                        if build_features:
                            pred_issues[id_] = self.build_features_dump(predictions, stack_ids, st_timestamps, ts)
                        else:
                            pred_issues[id_] = self.build_hist_ts(predictions, stack_ids, st_timestamps, ts)
                else:
                    for id_, issue in issues.items():
                        stack_ids = list(issue.stacks.keys())
                        predictions = self.stack_model.predict(st_id, stack_ids)
                        if build_features:
                            pred_issues[id_] = self.build_features_dump(predictions, stack_ids, st_timestamps, ts)
                        else:
                            pred_issues[id_] = self.build_hist_ts(predictions, stack_ids, st_timestamps, ts)

                event_dump = {"right": event.is_id, "issues": pred_issues}
                event_dump_str = json.dumps(event_dump)

                f.write(event_dump_str)
                f.write("\n")

    @staticmethod
    def build_hist_ts(predictions: List[float], stack_ids: List[int],
                      st_timestamps: Dict[int, int], ts: int) -> Dict[float, List[int]]:
        count_dict = {}
        for pred, stack in zip(predictions, stack_ids):
            cur_value = abs(st_timestamps[stack] - ts)
            count_dict[float(pred)] = count_dict.get(float(pred), [0, cur_value])
            count_dict[float(pred)][0] += 1
            count_dict[float(pred)][1] = min(count_dict[float(pred)][1], cur_value)
        return count_dict

    @staticmethod
    def build_features_dump(predictions: List[float], stack_ids: List[int],
                            st_timestamps: Dict[int, int], ts: int):
        if len(predictions) == 0:
            return ()

        scores = [float(score) for score in predictions]

        timestamps_diff = [abs(st_timestamps[stack] - ts) for stack in stack_ids]
        weights = 1.0 / (np.log(np.array(timestamps_diff) + 1) + 1)

        max_score_id = np.argmax(scores)

        first_max = scores[max_score_id]
        first_max_weight = float(weights[max_score_id])

        max_weight = float(max(weights))
        min_weight = float(min(weights))
        diff_max_min_weight = max_weight - min_weight
        mean_weight = float(np.mean(weights))

        hist = np.histogram(scores, bins=12, weights=weights)[0]
        weighted_hist = [float(h) for h in hist]

        hist = np.histogram(weights, bins=10)[0]
        weights_hist = [float(h) for h in hist]

        res = [first_max] + weighted_hist + [max_weight, min_weight, mean_weight] + \
              weights_hist + [diff_max_min_weight, first_max_weight]

        return tuple(res)
