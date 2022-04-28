import argparse
import os
import pickle
import warnings
from time import time

from src.common.utils import set_seed, random_seed
from src.similarity.data.buckets.bucket_data import BucketData, DataSegment, EventsBucketData
from src.similarity.data.dump_builder import DumpBuilder
from src.similarity.methods.pair_stack_issue_model import PairStackBasedSimModel
from src.similarity.scripts.similarity import add_data_arguments


def build_dump(data: BucketData, segment: DataSegment, ps_model: PairStackBasedSimModel, dump_path: str):
    start = time()
    dump_builder = DumpBuilder(ps_model.filter_model, ps_model.stack_model)
    dump_builder.dump_issue_scores_filtered(data.get_events(segment), dump_path, data.st_timestamps)

    print("Time to dump test", time() - start)


def read_args():
    parser = argparse.ArgumentParser(description='Dump')
    parser.add_argument('--data_start', type=int)
    parser.add_argument('--data_longitude', type=int, default=20)
    parser.add_argument('--dump_save_path', type=str, help='Where to save dump file')
    parser.add_argument('--model_path', type=str, help='Where to load similarity model')
    return add_data_arguments(parser).parse_args()


def dump():
    args = read_args()
    set_seed(random_seed)
    print('You passed next arguments to dump:')
    print('\n'.join(f'\t{k} = {v}' for k, v in vars(args).items()))

    os.makedirs(os.path.dirname(args.dump_path_save), exist_ok=True)

    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    segment = DataSegment(args.data_start, args.data_longitude)
    data = EventsBucketData(args.data_name, args.actions_file, args.reports_path, forget_days=args.forget_days)
    data.load()

    build_dump(data, segment, model, args.dump_save_path)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    dump()
