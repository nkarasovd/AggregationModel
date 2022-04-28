from typing import List

from src.similarity.data.buckets.bucket_data import BucketData, DataSegment
from src.similarity.data.buckets.event_state_model import StackAdditionEvent


class IssuesDataStats:
    def __init__(self, data: BucketData, train: DataSegment, val: DataSegment, test: DataSegment):
        self.data = data
        self.data.load()
        self.train = train
        self.val = val
        self.test = test

    def print_metrics(self):
        self.splited_metrics()

    def splited_metrics(self):
        print("Train")
        train_events = [e for e in self.data._time_slice_events(self.train.start, self.train.start + self.train.longitude)]
        self.issues_len(train_events)
        self.events_len(train_events)

        print("Val")
        val_events = [e for e in self.data._time_slice_events(self.val.start, self.val.start + self.val.longitude)]
        self.issues_len(val_events)
        self.events_len(val_events)

        print("Test")
        test_events = [e for e in self.data._time_slice_events(self.test.start, self.test.start + self.test.longitude)]
        self.issues_len(test_events)
        self.events_len(test_events)

    @staticmethod
    def issues_len(events: List[StackAdditionEvent]):
        issues = set()
        for event in events:
            issues.add(event.is_id)
        print("Issues num", len(issues))

    @staticmethod
    def events_len(events: List[StackAdditionEvent]):
        print("Events num", len(events))
