import json
import os
from argparse import ArgumentParser

import attr

from src.similarity.data.stack_loader import JsonStackLoader


@attr.s(auto_attribs=True, frozen=True)
class Event:
    st_id: int
    is_id: int
    ts: int
    label: bool


def convert_reports(state_path: str, reports_path: str):
    loader = JsonStackLoader(state_path)
    raw_reports = json.load(open(state_path, 'r'))

    os.makedirs(reports_path, exist_ok=True)
    for report in raw_reports:
        if report is None:
            continue

        st_id = int(report["bug_id"])
        report = loader(st_id)
        report.save_json(os.path.join(reports_path, f"{st_id}.json"))


def convert_events(state_path: str, action_path: str):
    actions = []
    raw_reports = json.load(open(state_path, 'r'))

    for report in raw_reports:
        if report is None:
            continue

        st_id = report["bug_id"]
        dup_id = report["dup_id"] or st_id
        ts = report["creation_ts"]

        action = Event(st_id, dup_id, ts, True)
        actions.append(action)

    actions = sorted(actions, key=lambda x: x.ts)

    os.makedirs(os.path.dirname(action_path), exist_ok=True)
    with open(action_path, 'w') as f:
        f.write("ts,rid,iid,label\n")
        for action in actions:
            f.write(",".join(map(str, [action.ts, action.st_id, action.is_id, action.label])))
            f.write("\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--state_path", type=str, help="Path to json file with Irving-formatted data presented "
                                                       "as a final system state")
    parser.add_argument("--reports_path", type=str, help="Path to directory with json files for every report")
    parser.add_argument("--events_path", type=str, help="Path to csv file with events of moving reports "
                                                        "between issues")
    args = parser.parse_args()

    convert_events(args.state_path, args.events_path)
    convert_reports(args.state_path, args.reports_path)


if __name__ == '__main__':
    main()
