import copy
from random import shuffle
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.aggregation.data.objects import LabeledQuery, LabeledQueryFeatures
from src.aggregation.data.scaler import Scaler
from src.aggregation.evaluation.prediction import predict
from src.aggregation.features.features_builder import FeaturesBuilder
from src.aggregation.features.features_config import FeaturesConfig
from src.aggregation.scorers.pairwise_rank_scorer import PairwiseRankScorer
from src.aggregation.scorers.rank_model.linear_rank_model import LinearRankModel
from src.aggregation.scorers.rank_model.rank_model import RankModel
from src.aggregation.train.train_log import BestResult
from src.common.evaluation import paper_metrics_iter


def _get_loss(rank_model: RankModel, train_query: LabeledQueryFeatures,
              device: torch.device, max_size: Optional[int]) -> Optional[torch.Tensor]:
    loss_function = nn.CrossEntropyLoss()
    right_issue_id = train_query.right_issue_id
    issues_features = train_query.issues_features

    bad = [tuple(fs) for id_, fs in issues_features.items() if id_ != right_issue_id]
    bad = [list(fs) for fs in set(bad)]

    if not bad:
        return None

    scores = rank_model.model(torch.tensor(bad, dtype=torch.float).to(device)).view(-1).cpu().tolist()
    bad_predictions = sorted(list(zip(scores, bad)), reverse=True)

    bad_examples = [b for _, b in bad_predictions[:max_size]]

    if bad_examples:
        bad_examples = torch.tensor(bad_examples, dtype=torch.float).to(device)

        good_examples = [issues_features[right_issue_id]] * len(bad_examples)
        good_examples = torch.tensor(good_examples, dtype=torch.float).to(device)

        target = torch.tensor([1] * len(bad_examples))

        predictions = rank_model(bad_examples, good_examples).cpu()

        return loss_function(predictions, target)


def _get_labeled_query_features(features_config: FeaturesConfig, q_train: List[LabeledQuery],
                                scaler: Scaler) -> List[LabeledQueryFeatures]:
    features_builder = FeaturesBuilder(features_config)
    train_queries = []
    for labeled_query in tqdm(q_train, position=0, leave=True):
        features = features_builder.get_features(labeled_query.query)
        issues_features = scaler.transform(features)
        train_queries.append(LabeledQueryFeatures(labeled_query.issue_id, issues_features))
    return train_queries


def train_pairwise_scorer(q_tr: List[LabeledQuery], q_te: List[LabeledQuery],
                          features_config: FeaturesConfig,
                          num_epochs: int = 250, scale: bool = False,
                          device: Optional[torch.device] = None) -> PairwiseRankScorer:
    device = device or torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rank_model = LinearRankModel(features_config.get_config_dim()).to(device)
    optimizer = Adam(rank_model.parameters())

    scaler = Scaler(scale)
    scaler.fit([q.query for q in q_tr], features_config)

    train_queries = _get_labeled_query_features(features_config, q_tr, scaler)

    evaluation_scorer = PairwiseRankScorer(rank_model, features_config, scaler, device)

    best_result = None

    for epoch in tqdm(range(num_epochs)):
        rank_model.train()
        shuffle(train_queries)

        for train_query in train_queries:
            optimizer.zero_grad()
            loss = _get_loss(rank_model, train_query, device, 10)
            if loss is not None:
                loss.backward()
                optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'\n\nEpoch = {epoch + 1}')
            rank_model.eval()
            scores = paper_metrics_iter(predict(evaluation_scorer, q_te))

            # Save new best result
            if best_result is None or scores['rr@1'][0] > best_result.scores['rr@1'][0]:
                best_result = BestResult(epoch, copy.deepcopy(rank_model), scores)

    best_result.print_log()

    return PairwiseRankScorer(best_result.rank_model, features_config, scaler, device)
