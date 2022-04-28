import argparse
import warnings

from src.aggregation.data.readers import read_data
from src.aggregation.features.features_config import FeaturesConfig, similarity_features, best_features
from src.aggregation.scorers.scorer import AggregationScorer
from src.aggregation.train.train_pairwise_scorer import train_pairwise_scorer
from src.aggregation.utils import timeit
from src.common.utils import set_seed, random_seed


@timeit
def train_aggregation_model(dump_path_train: str, dump_path_test: str,
                            features_config: FeaturesConfig) -> AggregationScorer:
    q_tr = read_data(dump_path_train)
    q_te = read_data(dump_path_test)

    print()
    return train_pairwise_scorer(q_tr, q_te, features_config)


def aggregation_parser():
    parser = argparse.ArgumentParser(description='Stack similarity aggregation')
    parser.add_argument('--train_path', type=str, help='Train similarities scores dump path')
    parser.add_argument('--test_path', type=str, help='Test similarities scores dump path')
    parser.add_argument('--model_save_path', type=str, help='Path to save rank model')
    return parser.parse_args()


def aggregation():
    set_seed(random_seed)
    args = aggregation_parser()
    print('You passed next arguments to aggregation:')
    print('\n'.join(f'\t{k} = {v}' for k, v in vars(args).items()))

    train_aggregation_model(args.train_path, args.test_path, similarity_features)

    pairwise_model = train_aggregation_model(args.train_path, args.test_path, best_features)

    pairwise_model.save(args.model_save_path)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    aggregation()
