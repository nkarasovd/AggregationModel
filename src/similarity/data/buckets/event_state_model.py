import os
import pickle
from typing import Iterable, Dict, Optional, Tuple, Set

import attr

from src.similarity.data.objects import Issue


@attr.s(auto_attribs=True, frozen=True)
class StackAdditionEvent:
    id: int
    st_id: int
    is_id: int
    ts: int
    label: bool


class StackAdditionState:
    def __init__(self, id: int, st_id: int, issues: Dict[int, Issue], is_id: int, label: bool = True):
        self.id = id
        self.st_id = st_id
        self.issues = issues
        self.is_id = is_id
        self.label = label

    @staticmethod
    def from_event(event: StackAdditionEvent, issues: Dict[int, Issue]):
        return StackAdditionState(event.id, event.st_id, issues, event.is_id, event.label)


class EventStateModel:
    def __init__(self, name: str, filter_label: bool, forget_days: Optional[float] = None):
        self.name = name
        self.issues: Dict[int, Issue] = {}
        self.stacks: Dict[int, Tuple[int, int]] = {}
        self.actual_issues: Set[int] = set()
        self.current_ts = None
        self.filter_label = filter_label
        self.forget_days = forget_days

    def update_state(self, addition_event: StackAdditionEvent):
        st_id = addition_event.st_id
        is_id = addition_event.is_id
        ts = addition_event.ts
        label = addition_event.label
        self.current_ts = ts

        if st_id in self.stacks:
            prev_ts, prev_is_id = self.stacks[st_id]
            self.issues[prev_is_id].remove(st_id, ts, addition_event.label)

        if is_id not in self.issues:
            self.issues[is_id] = Issue(is_id, ts)
        if self.filter_label and label or not self.filter_label:
            self.issues[is_id].add(st_id, ts, addition_event.label)
            self.stacks[st_id] = ts, is_id
        else:
            self.issues[is_id].update_ts(ts)
        self.actual_issues.add(is_id)

        if self.forget_days is not None and self.forget_days != 0:
            self.actual_issues = set(x for x in self.actual_issues if x != -1 and \
                                     self.current_ts - self.issues[x].last_ts() < self.forget_days)

    def warmup(self, actions: Iterable[StackAdditionEvent]):
        for action in actions:
            self.update_state(action)

    def collect(self, actions: Iterable[StackAdditionEvent]) -> Iterable[StackAdditionState]:
        for action in actions:
            if action.label and action.is_id != -1 and action.is_id != action.st_id:
                current_issues = {id: self.issues[id] for id in self.actual_issues}
                event = StackAdditionState.from_event(action, current_issues)
                yield event
            self.update_state(action)

    def all_seen_stacks(self) -> Iterable[int]:
        return self.stacks.keys()

    def file_name(self, days_num: float):
        if days_num == int(days_num):
            days_num = int(days_num)
        label = "_label" if self.filter_label else ""
        return f'../event_states/{self.name}_event_state_{days_num}{label}.pickle'

    def load(self, days_num: float):
        with open(self.file_name(days_num), 'rb') as f:
            tmp_dict = pickle.load(f)

        for data_field in ['issues', 'stacks', 'actual_issues', 'current_ts', 'forget_days']:
            self.__dict__[data_field] = tmp_dict[data_field]
        # self.__dict__.update(tmp_dict)

    def save(self, days_num: float):
        file_path = self.file_name(days_num)
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
