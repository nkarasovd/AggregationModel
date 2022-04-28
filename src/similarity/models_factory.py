from typing import List

from src.similarity.data.stack_loader import StackLoader
from src.similarity.methods.base import SimStackModel
from src.similarity.methods.classic.crash_graphs import CrashGraphs
from src.similarity.methods.classic.durfex import DurfexModel
from src.similarity.methods.classic.lerch import LerchModel
from src.similarity.methods.classic.levenshtein import LevenshteinModel
from src.similarity.methods.classic.trace_sim import TraceSimModel
from src.similarity.methods.neural import device
from src.similarity.methods.neural.neural_base import NeuralModel
from src.similarity.methods.neural.siam.aggregation import ConcatAggregation
from src.similarity.methods.neural.siam.encoders import LSTMEncoder
from src.similarity.methods.neural.siam.siam_network import SiamMultiModalModel
from src.similarity.preprocess.entry_coders import Exception2Seq, Stack2Seq, MultiEntry2Seq
from src.similarity.preprocess.seq_coder import SeqCoder
from src.similarity.preprocess.tokenizers import SimpleTokenizer


def create_classic_model(stack_loader: StackLoader, method: str = 'lerch',
                         use_ex: bool = False, max_len: int = None,
                         trim_len: int = 0, sep: str = '.') -> SimStackModel:
    stack2seq = Stack2Seq(cased=False, trim_len=trim_len, sep=sep)
    if use_ex:
        stack2seq = MultiEntry2Seq([stack2seq, Exception2Seq(cased=False, trim_len=trim_len, throw=False, to_set=True)])
    coder = SeqCoder(stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=max_len)
    if method == 'lerch':
        model = LerchModel(coder)
    elif method == 'tracesim':
        model = TraceSimModel(coder)
    elif method == 'levenshtein':
        model = LevenshteinModel(coder)
    elif method == 'durfex':
        model = DurfexModel(coder, ns=(1, 2, 3))
    elif method == 'crash_graphs':
        model = CrashGraphs(coder)
    else:
        raise ValueError("Method name is not match")
    return model


def create_neural_model(stack_loader: StackLoader, unsup_data: List[int],
                        use_ex: bool = False, max_len: int = None, trim_len: int = 0, sep: str = '.') -> NeuralModel:
    stack2seq = MultiEntry2Seq([Stack2Seq(cased=False, trim_len=trim_len, sep=sep),
                                Exception2Seq(cased=False, trim_len=trim_len, throw=False, to_set=True)]) \
        if use_ex else Stack2Seq(cased=False, trim_len=trim_len, sep=sep)

    coder = SeqCoder(stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=max_len)

    coder.fit(unsup_data)

    encoders = [LSTMEncoder(coder, dim=50, hid_dim=100).to(device)]
    model = SiamMultiModalModel(encoders, ConcatAggregation, features_num=4, out_num=1).to(device)

    model.to(device)

    return model
