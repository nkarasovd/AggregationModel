import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Iterable

import numpy as np
import pandas as pd
import attr

from src.similarity.data.buckets.event_state_model import StackAdditionState, EventStateModel, StackAdditionEvent
from src.similarity.data.stack_loader import StackLoader, DirectoryStackLoader


@attr.s(auto_attribs=True, frozen=True)
class DataSegment:
    start: int
    longitude: int


class BucketData(ABC):
    def __init__(self, name: str, sep: str = '.', filter_label: bool = False, forget_days: Optional[float] = None):
        self.name = name
        self.sep = sep
        self.filter_label = filter_label
        self.forget_days = forget_days

        self.actions = []
        self.st_timestamps = {}

        self._all_reports = {}

    @abstractmethod
    def load(self) -> 'BucketData':
        raise NotImplementedError

    @abstractmethod
    def stack_loader(self) -> StackLoader:
        raise NotImplementedError

    def _time_slice_events(self, start: float, finish: float) -> List[StackAdditionEvent]:
        return [event for event in self.actions if start <= event.ts < finish]

    def _cached_event_state(self, until_day: float) -> EventStateModel:
        event_model = EventStateModel(self.name, self.filter_label, self.forget_days)

        if os.path.exists(event_model.file_name(until_day)):
            event_model.load(until_day)
        else:
            load_prev = False
            for i in range(int(until_day), 0, -1):
                if os.path.exists(event_model.file_name(i)):
                    event_model.load(i)
                    event_model.warmup(self._time_slice_events(i, until_day))
                    load_prev = True
                    print(f"post train from {i} to {until_day}")
                    break
            if not load_prev:
                event_model.warmup(self._time_slice_events(0, until_day))

            event_model.save(until_day)

        return event_model

    def _generate_events(self, start: float, longitude: float) -> Iterable[StackAdditionState]:
        event_model = self._cached_event_state(start)
        return event_model.collect(self._time_slice_events(start, start + longitude))

    def all_reports(self, data_segment: DataSegment) -> List[int]:
        start = data_segment.start
        longitude = data_segment.longitude

        if (start, longitude) not in self._all_reports:
            event_model = self._cached_event_state(start + longitude)
            reports = sorted(list(event_model.all_seen_stacks()))
            self._all_reports[(start, longitude)] = reports
        return self._all_reports[(start, longitude)]

    def get_events(self, data_segment: DataSegment) -> Iterable[StackAdditionState]:
        start = data_segment.start
        longitude = data_segment.longitude

        return self._generate_events(start, longitude)


class EventsBucketData(BucketData):
    def __init__(self, name: str, actions_file: str, reports_path: str, other_report_dirs: List[str] = None,
                 forget_days: Optional[float] = None, filter_label: bool = False):
        super().__init__(name, sep='.', filter_label=filter_label, forget_days=forget_days)
        self.reports_path = reports_path
        self.actions_file = actions_file
        self.other_report_dirs = other_report_dirs or []
        self.raw_actions = None
        self.st_issue = None

    def load(self) -> 'EventsBucketData':
        self.raw_actions = pd.read_csv(self.actions_file).values
        first_ts = datetime.fromtimestamp(self.raw_actions[0, 0])
        day_secs = 60 * 60 * 24
        self.raw_actions[:, 0] = np.array([(datetime.fromtimestamp(a[0]) - first_ts).total_seconds()
                                           / day_secs for a in self.raw_actions])
        self.raw_actions = np.hstack(
            (np.array(range(len(self.raw_actions))).reshape(-1, 1), self.raw_actions))

        self.actions = [
            StackAdditionEvent(id_, st_id, is_id, ts, label) for id_, ts, st_id, is_id, label in self.raw_actions
        ]
        if self.filter_label:
            self.actions = [action for action in self.actions if action.label]

        self.st_issue = {action.st_id: action.is_id for action in self.actions}
        self.st_timestamps = {action.st_id: int(action.ts) for action in self.actions}

        return self

    def stack_loader(self) -> StackLoader:
        return DirectoryStackLoader(self.reports_path, *self.other_report_dirs, st_issue=self.st_issue)
