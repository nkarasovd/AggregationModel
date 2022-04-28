import argparse
from typing import Optional

from src.similarity.data.buckets.bucket_data import DataSegment
from src.similarity.data.buckets.bucket_data import EventsBucketData
from src.similarity.evaluation.data_metrics import IssuesDataStats
from src.similarity.scripts.similarity import add_data_arguments


def netbeans_info(data_name: str, actions_file: str, reports_path: str, forget_days: Optional[float]):
    netbeans_val = DataSegment(4200, 140)
    netbeans_test = DataSegment(4340, 350)

    data = EventsBucketData(data_name, actions_file, reports_path, forget_days=forget_days)
    data_info = IssuesDataStats(data, DataSegment(350, 3850), netbeans_val, netbeans_test)
    data_info.print_metrics()

    print()

    data_info = IssuesDataStats(data, DataSegment(350, 3650), netbeans_val, netbeans_test)
    data_info.print_metrics()

    print()

    data_info = IssuesDataStats(data, DataSegment(4000, 200), netbeans_val, netbeans_test)
    data_info.print_metrics()


def main():
    args = add_data_arguments(argparse.ArgumentParser()).parse_args()
    netbeans_info(args.data_name, args.actions_file, args.reports_path, args.forget_days)


if __name__ == '__main__':
    main()
