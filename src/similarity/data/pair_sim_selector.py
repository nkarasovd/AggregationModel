from abc import ABC
from typing import Tuple, List, Iterable

import numpy as np

from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.methods.hypothesis_selection import HypothesisSelector


class PairSimSelector(ABC):
    def __call__(self, event: StackAdditionState) -> Tuple[List[int], List[int]]:
        raise NotImplementedError

    def generate(self, events: List[StackAdditionState]) -> Iterable[Tuple[int, int, int]]:
        for event in events:
            st_id = event.st_id
            try:
                st_ids, target = self(event)
                for sid, sim in zip(st_ids, target):
                    yield st_id, sid, sim
            except:
                pass


class RandomPairSimSelector(PairSimSelector):
    def __init__(self, size: int = None):
        self.size = size

    def __call__(self, event: StackAdditionState) -> Tuple[List[int], List[int]]:
        good_stacks = np.random.permutation([(sid, 1) for sid in event.issues[event.is_id].stacks])[:self.size]
        bad_issues = np.random.permutation(list(set(event.issues.keys() - {event.is_id})))
        bad_stacks = []
        for iid in bad_issues:
            istacks = [[sid, 0] for sid in event.issues[iid].stacks]
            if not istacks:
                continue
            bad_stacks.append(istacks[np.random.randint(len(istacks))])
            if len(bad_stacks) >= self.size:
                break
        stacks = np.vstack([good_stacks, np.array(bad_stacks)])
        stacks = np.random.permutation(stacks)[:self.size]
        return tuple(map(list, zip(*stacks)))


class TopPairSimSelector(PairSimSelector):
    def __init__(self, hypothesis_selector: HypothesisSelector, size: int = None):
        self.hypothesis_selector = hypothesis_selector
        self.size = size

    def __call__(self, event: StackAdditionState) -> Tuple[List[int], List[int]]:
        top_issues_stacks = self.hypothesis_selector.filter_top(event.id, event.st_id, event.issues)

        if event.is_id in top_issues_stacks:
            good_stacks = top_issues_stacks[event.is_id]
        elif event.is_id in event.issues:
            good_stacks = list(event.issues[event.is_id].stacks.keys())
        else:
            # it's new issue!
            good_stacks = []
        good_stacks = np.random.permutation([(sid, 1) for sid in good_stacks])

        bad_stacks = []
        for k, vs in top_issues_stacks.items():
            if k != event.is_id:
                bad_stacks += vs
        size = min(self.size, len(good_stacks), len(bad_stacks))
        if size == 0:
            return [], []
        good_stacks = good_stacks[:size]
        bad_stacks = np.random.permutation([(id, 0) for id in bad_stacks])[:size]

        stacks = np.vstack([good_stacks, bad_stacks])
        stacks = np.random.permutation(stacks)[:self.size]
        return tuple(map(list, zip(*stacks)))
