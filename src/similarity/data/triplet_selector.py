from abc import ABC
from typing import Tuple, List, Iterable

import numpy as np

from src.similarity.data.buckets.event_state_model import StackAdditionState
from src.similarity.methods.hypothesis_selection import HypothesisSelector


class TripletSelector(ABC):
    def __call__(self, event: StackAdditionState) -> Tuple[List[int], List[int]]:
        raise NotImplementedError

    def generate(self, events: List[StackAdditionState]) -> Iterable[Tuple[int, int, int]]:
        for event in events:
            try:
                st_ids, target = self.__call__(event)
                for sid, sim in zip(st_ids, target):
                    yield event.st_id, sid, sim
            except:
                pass


class RandomTripletSelector(TripletSelector):
    def __init__(self, size: int = None):
        self.size = size

    def __call__(self, event: StackAdditionState) -> Tuple[List[int], List[int]]:
        """
        Return list of good stack ids and bad stack ids

        :param issues: dict of issue_id to issue
        :param is_id: the issue new stack was attached
        :return: list of good stack ids and bad stack ids
        """
        if event.is_id not in event.issues:
            return [], []
        good_stacks = np.array(list(event.issues[event.is_id].stacks.keys()))
        bad_stacks = []
        for iid in event.issues.keys() - {event.is_id}:
            bad_stacks += list(event.issues[iid].stacks.keys())

        good_stacks = good_stacks[np.random.choice(len(good_stacks), self.size)]
        bad_stacks = np.array(bad_stacks)[np.random.choice(len(bad_stacks), self.size)]

        return list(good_stacks), list(bad_stacks)


class TopTripletSelector(TripletSelector):
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

        bad_stacks = []
        for k, vs in top_issues_stacks.items():
            if k != event.is_id:
                bad_stacks += vs

        size = min(self.size, len(good_stacks), len(bad_stacks))
        good_stacks = np.array(np.random.permutation(good_stacks)[:size])
        bad_stacks = np.array(np.random.permutation(bad_stacks)[:size])

        return list(good_stacks), list(bad_stacks)
