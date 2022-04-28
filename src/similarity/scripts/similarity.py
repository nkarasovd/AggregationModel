import argparse
import os
import pickle
import warnings
from typing import Optional

from src.common.utils import set_seed, random_seed
from src.similarity.data.buckets.bucket_data import DataSegment, EventsBucketData
from src.similarity.methods.pair_stack_issue_model import PairStackBasedSimModel
from src.similarity.scripts.train_similarity_model import train_classic_model, train_neural_model


def train_similarity_model(method: str,
                           train_start, train_longitude,
                           val_start, val_longitude,
                           test_start, test_longitude,
                           data_name: str, actions_file: str, reports_path: str,
                           forget_days: Optional[int],
                           hyp_top_issues: Optional[int], hyp_top_reports: Optional[int],
                           filter_thr: Optional[float],
                           filter_label: bool
                           ):
    train = DataSegment(train_start, train_longitude)
    val = DataSegment(val_start, val_longitude)
    test = DataSegment(test_start, test_longitude)

    data = EventsBucketData(data_name, actions_file, reports_path,
                            forget_days=forget_days, filter_label=filter_label)
    data.load()

    if method == 's3m':
        return train_neural_model(data, train, val, test,
                                  unsup=False, use_ex=False, max_len=None, trim_len=0,
                                  filter_thr=filter_thr,
                                  hyp_top_issues=hyp_top_issues, hyp_top_stacks=hyp_top_reports,
                                  loss_name='ranknet', epochs=5
                                  )
    else:
        return train_classic_model(data, train, val, test, method, trim_len=0,
                                   filter_thr=filter_thr,
                                   hyp_top_issues=hyp_top_issues, hyp_top_stacks=hyp_top_reports,
                                   )


def save_model(similarity_model: PairStackBasedSimModel, model_path_save: str):
    if model_path_save is not None:
        os.makedirs(os.path.dirname(model_path_save), exist_ok=True)
        with open(model_path_save, 'wb') as f:
            pickle.dump(similarity_model, f)


def add_data_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--data_name', type=str, help='Data name')
    parser.add_argument('--actions_file', type=str, help='Events actions file path')
    parser.add_argument('--reports_path', type=str, help='Directory with json reports')
    parser.add_argument('--forget_days', type=int, default=None)
    return parser


def read_args():
    parser = argparse.ArgumentParser(description='Similarity model')
    parser.add_argument('--seed', type=int, required=False, default=None)
    parser.add_argument('--method', type=str,
                        default='lerch',
                        choices=['lerch', 'tracesim', 'moroo',
                                 'rebucket', 'cosine', 'levenshtein',
                                 'brodie', 'prefix', 's3m', 'durfex', 'crash_graphs'],
                        help='Method for similarity prediction')

    parser.add_argument('--train_start', type=int)
    parser.add_argument('--train_longitude', type=int, default=80)

    parser.add_argument('--val_start', type=int)
    parser.add_argument('--val_longitude', type=int, default=7)

    parser.add_argument('--test_start', type=int)
    parser.add_argument('--test_longitude', type=int, default=20)

    parser.add_argument('--hyp_top_issues', type=int, required=False, default=None)
    parser.add_argument('--hyp_top_reports', type=int, required=False, default=None)
    parser.add_argument('--filter_thr', type=float, required=False, default=None)

    parser.add_argument('--filter_label', action="store_true", default=False)

    parser.add_argument('--model_save_path', type=str, help='Where to save similarity model')

    parser = add_data_arguments(parser)

    args = parser.parse_args()

    print('You passed next arguments to similarity:')
    print('\n'.join(f'\t{k} = {v}' for k, v in vars(args).items()))
    if args.model_save_path is None:
        print("Model will is not be saved, because model_save_path is not specified")

    return args


def similarity():
    args = read_args()
    set_seed(args.seed or random_seed)

    similarity_model = train_similarity_model(
        args.method,
        args.train_start, args.train_longitude,
        args.val_start, args.val_longitude,
        args.test_start, args.test_longitude,
        args.data_name, args.actions_file, args.reports_path,
        args.forget_days,
        args.hyp_top_issues, args.hyp_top_reports,
        args.filter_thr,
        args.filter_label,
    )

    save_model(similarity_model, args.model_save_path)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    similarity()
