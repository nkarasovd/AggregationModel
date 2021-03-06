import math
from typing import List, Tuple, Dict

from src.similarity.methods.classic.hyper_opt import SimStackHyperoptModel
from src.similarity.methods.classic.levenshtein import levenshtein_dist
from src.similarity.methods.classic.tfidf import IntTfIdfComputer
from src.similarity.preprocess.seq_coder import SeqCoder


class TraceSimModel(SimStackHyperoptModel):
    def __init__(self, coder: SeqCoder = None, alpha: float = None, beta: float = None, gamma: float = None):
        self.tfidf = IntTfIdfComputer(coder)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def params_edges(self) -> Dict[str, Tuple[float, float]]:
        return {
            "alpha": (0, 0.5),
            "beta": (0, 5),
            "gamma": (0, 15)
        }

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, unsup_data: List[int] = None) -> 'TraceSimModel':
        self.tfidf.fit(unsup_data)
        return self

    def weights(self, stack_id: int, coded_seq: List[int], alpha: float, beta: float, gamma: float) -> List[float]:
        local_weight = [1 / (1 + i) ** alpha for i, _ in enumerate(coded_seq)]

        idfs = self.tfidf.transform(stack_id)
        global_weight = []
        for word in coded_seq:
            tf, idf = idfs.get(word, (0, 0))
            score = idf
            global_weight.append(1 / (1 + math.exp(-beta * (score - gamma))))

        return [lw * gw for lw, gw in zip(local_weight, global_weight)]

    def predict(self, anchor_id: int, stack_ids: List[int],
                alpha: float = None, beta: float = None, gamma: float = None) -> List[float]:
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma

        scores = []
        anchor_seq = self.tfidf.coder(anchor_id)
        anchor_weights = self.weights(anchor_id, anchor_seq, alpha, beta, gamma)
        for stack_id in stack_ids:
            stack_seq = self.tfidf.coder(stack_id)
            stack_weights = self.weights(stack_id, stack_seq, alpha, beta, gamma)
            max_dist = sum(anchor_weights) + sum(stack_weights)
            dist = levenshtein_dist(anchor_seq, anchor_weights, stack_seq, stack_weights)
            score = 0 if max_dist == 0 else 1 - dist / max_dist
            scores.append(score)
        return scores

    def name(self) -> str:
        return self.tfidf.name() + f"_tracesim_{self.alpha}_{self.beta}_{self.gamma}"
