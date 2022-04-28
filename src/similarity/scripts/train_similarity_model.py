from time import time
from typing import List
from typing import Optional

import torch

from src.common.evaluation import paper_metrics_iter
from src.common.utils import set_seed, random_seed
from src.similarity.data.buckets.bucket_data import BucketData, DataSegment
from src.similarity.data.stack_loader import StackLoader
from src.similarity.methods.classic.hyper_opt import PairStackBasedIssueHyperoptModel
from src.similarity.methods.classic.opt_lerch import OptLerchModel
from src.similarity.methods.hypothesis_selection import HypothesisSelector, TopNumHypothesisSelector, \
    ScoreThrHypothesisSelector, TopIssuesHypothesisSelector
from src.similarity.methods.neural.train_issue_sim import train_issue_model
from src.similarity.methods.pair_stack_issue_model import MaxIssueScorer, PairStackBasedSimModel
from src.similarity.models_factory import create_neural_model, create_classic_model
from src.similarity.preprocess.entry_coders import Stack2Seq
from src.similarity.preprocess.seq_coder import SeqCoder
from src.similarity.preprocess.tokenizers import SimpleTokenizer


def hyp_selector(stack_loader: StackLoader, unsup_stacks: List[int], filter_thr: Optional[float],
                 hyp_top_stacks: Optional[int], hyp_top_issues: Optional[int],
                 sep: str
                 ) -> HypothesisSelector:
    assert int(filter_thr is not None) + int(hyp_top_stacks is not None or hyp_top_issues is not None) == 1

    start = time()
    coder = SeqCoder(stack_loader, Stack2Seq(cased=False, sep=sep), SimpleTokenizer(),
                     min_freq=0, max_len=None)

    lerch_model = OptLerchModel(coder).fit([], unsup_stacks)
    if filter_thr is not None:
        filter_model = ScoreThrHypothesisSelector(lerch_model, MaxIssueScorer(), thr=filter_thr)
    elif hyp_top_stacks is not None and hyp_top_issues is not None:
        filter_model = TopNumHypothesisSelector(lerch_model, MaxIssueScorer(), top_stacks=hyp_top_stacks,
                                                top_issues=hyp_top_issues)
    else:
        filter_model = TopIssuesHypothesisSelector(lerch_model, MaxIssueScorer(), top_issues=hyp_top_issues)

    print("Time to fit hyp selector model", time() - start)
    return filter_model


def print_data_info(data: BucketData, train: DataSegment, val: DataSegment, test: DataSegment,
                    loss_name: Optional[str] = None):
    print("Data:", data.name)
    print(f"{'%-19s' % 'Train dataset:'} start - {'%5d' % train.start}, longitude - {'%5d' % train.longitude}")
    print(f"Validation dataset: start - {'%5d' % val.start}, longitude - {'%5d' % val.longitude}")
    print(f"{'%-19s' % 'Test dataset:'} start - {'%5d' % test.start}, longitude - {'%5d' % test.longitude}")

    if loss_name:
        print(f"Loss name - {loss_name}")


def train_classic_model(data: BucketData, train: DataSegment, val: DataSegment, test: DataSegment,
                        method: str,
                        use_ex: bool = False, max_len: int = None, trim_len: int = 0,
                        filter_thr: Optional[float] = None,
                        hyp_top_stacks: Optional[int] = None, hyp_top_issues: Optional[int] = None
                        ) -> PairStackBasedIssueHyperoptModel:
    set_seed(random_seed)

    print_data_info(data, train, val, test)
    stack_loader = data.stack_loader()
    unsup_stacks = data.all_reports(train)
    print("Get train stacks")

    if filter_thr is not None or hyp_top_stacks is not None or hyp_top_issues is not None:
        filter_model = hyp_selector(stack_loader, unsup_stacks, filter_thr, hyp_top_stacks, hyp_top_issues, data.sep)
    else:
        filter_model = None
    print("Filter model:", filter_model)

    start = time()

    model = create_classic_model(stack_loader, method, use_ex, max_len, trim_len, data.sep)
    print("Create model")
    model.fit([], unsup_stacks)
    print("Fit model")
    ps_model = PairStackBasedIssueHyperoptModel(model, MaxIssueScorer(), filter_model)
    ps_model.find_params(data, val)
    print("Find params")
    print("Time to fit", time() - start)

    evaluate_model(data, test, ps_model)

    return ps_model


def train_neural_model(data: BucketData, train: DataSegment, val: DataSegment, test: DataSegment,
                       unsup: bool = False, use_ex: bool = False, max_len: int = None, trim_len: int = 0,
                       loss_name: str = 'point',
                       filter_thr: Optional[float] = None,
                       hyp_top_stacks: Optional[int] = None, hyp_top_issues: Optional[int] = None,
                       epochs: int = 1) -> PairStackBasedSimModel:
    set_seed(random_seed)

    print_data_info(data, train, val, test, loss_name)
    stack_loader = data.stack_loader()
    unsup_stacks = data.all_reports(train)

    model = create_neural_model(stack_loader, unsup_stacks, use_ex, max_len, trim_len)

    if filter_thr is not None or hyp_top_stacks is not None or hyp_top_issues is not None:
        filter_model = hyp_selector(stack_loader, unsup_stacks, filter_thr, hyp_top_stacks, hyp_top_issues, data.sep)
    else:
        filter_model = None
    print("Filter model:", filter_model)

    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)]
    train_issue_model(model, data, train, test, loss_name, optimizers, filter_model,
                      epochs=epochs, batch_size=10, period=100, selection_from_event_num=4, writer=None)

    ps_model = PairStackBasedSimModel(model, MaxIssueScorer(), filter_model)

    return ps_model


def evaluate_model(data: BucketData, test: DataSegment, ps_model: PairStackBasedSimModel):
    start = time()
    new_preds = ps_model.predict(data.get_events(test))
    print("Time to predict", time() - start)

    start = time()
    paper_metrics_iter(new_preds)
    print("Time to eval", time() - start)
